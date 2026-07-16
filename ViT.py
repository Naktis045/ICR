import os
# Suppress TF logs and symlink warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Dense, Embedding, LayerNormalization, MultiHeadAttention, Flatten
from datasets import load_dataset


#CONFIGURATION & HYPERPARAMETERS
CONFIGURATION = {
    "PATCH_SIZE": 16,
    "NUM_CLASSES": 80, 
}

TARGET_HEIGHT = 256
TARGET_WIDTH = 256
CHANNELS = 3  
BATCH_SIZE = 16  
AUTOTUNE = tf.data.experimental.AUTOTUNE


# LOAD DATASET
print("Loading Teklia/IAM-line dataset...")
hf_dataset = load_dataset("Teklia/IAM-line")

# Extract character set to build vocabulary dynamically
print("Building vocabulary from dataset...")
all_texts = list(hf_dataset['train']['text']) + list(hf_dataset['test']['text'])
vocab = sorted(list(set([text[0] for text in all_texts if len(text) > 0])))
char_to_num = {char: idx for idx, char in enumerate(vocab)}
CONFIGURATION["NUM_CLASSES"] = len(vocab) + 1

def tokenize_label(text):
    if len(text) == 0:
        return 0
    return char_to_num.get(text[0], 0)


# MEMORY-EFFICIENT GENERATORS
def train_gen():
    for item in hf_dataset['train']:
        img_np = np.array(item['image'].convert('RGB'))
        label = tokenize_label(item['text'])
        yield img_np, label

def test_gen():
    for item in hf_dataset['test']:
        img_np = np.array(item['image'].convert('RGB'))
        label = tokenize_label(item['text'])
        yield img_np, label


# PREPROCESSING & AUGMENTATION PIPELINES
@tf.function
def prepare_image(image, label):
    image = tf.image.resize(image, [TARGET_HEIGHT, TARGET_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

# Augmentation layers
random_rotation_layer = layers.RandomRotation(factor=5/360, fill_mode="constant", fill_value=1.0)
random_zoom_layer = layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.1, 0.1), fill_mode="constant", fill_value=1.0)
random_brightness_layer = layers.RandomBrightness(factor=0.15)
random_contrast_layer = layers.RandomContrast(factor=(0.8, 1.2))

@tf.function
def augment_image(image, label):
    image = random_rotation_layer(image, training=True)
    image = random_zoom_layer(image, training=True)
    image = random_brightness_layer(image, training=True)
    image = random_contrast_layer(image, training=True)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Build tf.data pipelines from generators
ds_train = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
ds_train = ds_train.map(prepare_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.shuffle(buffer_size=128) 
ds_train = ds_train.map(augment_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = tf.data.Dataset.from_generator(
    test_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)
ds_test = ds_test.map(prepare_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)


# CUSTOM VISION TRANSFORMER (ViT) MODEL

class Patch_Encoder(Layer):
    def __init__(self, num_patches, hidden_size, **kwargs):
        super(Patch_Encoder, self).__init__(name='patch_encoder', **kwargs)
        self.linear = Dense(hidden_size)
        self.positional_embedding = Embedding(input_dim=num_patches, output_dim=hidden_size)
        self.num_patches = num_patches

    def call(self, x):
        # Extract patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
            strides=[1, CONFIGURATION["PATCH_SIZE"], CONFIGURATION["PATCH_SIZE"], 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape patches dynamically
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, (batch_size, self.num_patches, -1))
        
        # Add positional embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        output = self.linear(patches) + self.positional_embedding(positions)
        return output


class Transformer_Encoder(Layer):
    def __init__(self, num_heads, hidden_size, **kwargs):
        super(Transformer_Encoder, self).__init__(**kwargs)
        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.dense_1 = Dense(hidden_size, activation=tf.nn.gelu)
        self.dense_2 = Dense(hidden_size, activation=tf.nn.gelu)

    def call(self, input_tensor):
        x_1 = self.layer_norm_1(input_tensor)
        x_1 = self.multi_head_attention(x_1, x_1)
        x_1 = layers.add([x_1, input_tensor]) 

        x_2 = self.layer_norm_2(x_1)
        x_2 = self.dense_1(x_2)
        output = self.dense_2(x_2)
        output = layers.add([output, x_1]) 
        return output


class VIT(Model):
    def __init__(self, num_heads, hidden_size, num_patches, num_dense_units, num_layers, **kwargs):
        super(VIT, self).__init__(name="VIT", **kwargs)
        self.num_layers = num_layers
        self.patch_encoder = Patch_Encoder(num_patches, hidden_size)
        self.tr_encoders = [Transformer_Encoder(num_heads, hidden_size, name=f"transformer_encoder_{i}") for i in range(num_layers)]
        self.dense_1 = Dense(num_dense_units, tf.nn.gelu) 
        self.dense_2 = Dense(num_dense_units, tf.nn.gelu) 
        self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation="softmax")

    def call(self, x, training=True):
        x = self.patch_encoder(x)
        for i in range(self.num_layers):
            x = self.tr_encoders[i](x)

        x = Flatten()(x)
        x = self.dense_1(x) 
        x = self.dense_2(x)
        return self.dense_3(x)


# INITIALIZE & BUILD MODEL CONFIGURATION

print("Building ViT model...")
vit = VIT(num_heads=8, hidden_size=768, num_patches=256, num_layers=4, num_dense_units=1024)

vit.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# FIXED: Instead of vit.build(), pass a dummy tensor to properly build the inner weights
dummy_input = tf.zeros((1, TARGET_HEIGHT, TARGET_WIDTH, CHANNELS))
_ = vit(dummy_input)

# Display a populated model summary
vit.summary()


# CALLBACKS CONFIGURATION
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='best_vit_weights.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

callbacks_list = [early_stopping, checkpoint_callback]


# TRAIN & EVALUATE
print("\nStarting model training...")
vit.fit(
    ds_train,
    validation_data=ds_test,
    epochs=30,
    callbacks=callbacks_list
)

print("\nEvaluating on Test Data:")
vit.evaluate(ds_test)

# Save the final trained weights cleanly
vit.save_weights('ViT_model_weights.h5')
print("Model training complete and weights saved successfully!")