import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib
from data_preprocessing import get_data

from model_custom import get_custom_model


def show_images(ds):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


if __name__ == '__main__':
    #GPU Custom Configs
    print(tf.version)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # Move Covid and Normal image folder to new train folder
    dataset_path = "../COVID-19_Radiography_Dataset/train"
    batch_size = 32
    img_height = 256
    img_width = 256

    data_dir = pathlib.Path(dataset_path)
    train_ds, val_ds = get_data(data_dir)

    # 1. Explore Data
    class_names = train_ds.class_names
    # print(class_names)
    show_images(train_ds)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation
    data_augmentation_layer = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    # Normalization layer
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

    model_custom = get_custom_model(data_augmentation_layer, normalization_layer)

    # Define Callback
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model_custom.compile(optimizer='adam',
                         loss=tf.keras.losses.binary_crossentropy,
                         metrics=['accuracy'])
    history = model_custom.fit(
      train_ds,
      validation_data=val_ds,
      epochs=50,
      callbacks=[es_callback]
    )

    model_custom.save('tl_model_tf.h5', save_format='tf')