import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tf import keras
from tf.keras import layers, regularizers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


(ds_train, ds_test), ds_info =tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info = True,
)

@tf.function
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

# Define the rotation layer (allows rotation between -25 and +25 degrees)
random_rotation_layer = tf.keras.layers.RandomRotation(factor=25/360)

@tf.function
def augment_image(image, label):
# Apply it to your image
    image = random_rotation_layer(image)
    image = tf.image.resize(image, [28, 28])
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.2)
     # Convert image to gray scale
    if tf.random.uniform((), minval=0, maxval=1)< 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image),[1,1,3])# Grayscale 1 channel(256, 256, 3),(256, 256, 1) - Saving disk space; models designed specifically for 1-channel inputs.
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    return image, label

AUTOTUNE=tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
# For train set
ds_train = ds_train.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)
#For test set
ds_test = ds_test.map(normalize_image, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)

def model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation ='relu', kernel_regularizer=regularizers.l2(0.01), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.01), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    training_data=ds_train,
    epochs=10,
    validation_data=ds_test,
    batch_size=32,
    verbose=2,
)
model.evaluate(ds_train, ds_test, batch_size=32, verbose=2)
model.save('emnist_model.h5')