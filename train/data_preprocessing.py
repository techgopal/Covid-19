import tensorflow as tf


def get_data(folder_path, batch_size=32, img_height=256, img_width=256):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        folder_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds
