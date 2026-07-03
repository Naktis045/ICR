import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_datasets as tfds
import math

# 1. Load Dataset (EMNIST Balanced has 47 classes)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
NUM_CLASSES = ds_info.features['label'].num_classes

@tf.function
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

# Mild augmentation suited for letters/numbers
random_rotation_layer = layers.RandomRotation(factor=10/360)
random_zoom_layer = layers.RandomZoom(height_factor=0.1, width_factor=0.1)

@tf.function
def augment_image(image, label):
    # CRITICAL FIX: Explicitly passing training=True makes sure augmentation runs!
    image = random_rotation_layer(image, training=True)
    image = random_zoom_layer(image, training=True)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64

# Process Train Set
ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.map(augment_image, num_parallel_calls=AUTOTUNE) 
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Process Test Set
ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)


# 2. Upgraded Deep CNN Architecture
def build_model():
    inputs = keras.Input(shape=(28, 28, 1))
    l2_reg = regularizers.l2(1e-5) # Fine-tuned weight penalty
    
    # Block 1
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x) 
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D()(x) 
    
    # Block 3 (Preserves fine text curves before pooling)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Classifier Head
    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=l2_reg)(x) 
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES)(x) 
    return keras.Model(inputs=inputs, outputs=outputs)

model = build_model()

# 3. Learning Rate Scheduler Callback (Fixed to return clean python float)
def lr_decay(epoch, lr):
    if epoch < 8:
        return float(lr)
    else:
        return float(lr * math.exp(-0.1)) # Uses standard python math instead of tf.math

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), # Higher initial speed
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 4. Fit and Save
model.fit(
    ds_train,
    epochs=20, # Expanded time frame to find a better global minimum
    validation_data=ds_test,
    callbacks=[lr_callback],
    verbose=2,
)

print("\nEvaluating on Test Data:")
model.evaluate(ds_test, verbose=2)

model.save('emnist_model.h5')
