import tensorflow as tf
import os

def load_data(raw_dir, reference_dir, image_size=(256, 256)):
    """Loads both raw underwater images and reference images for supervised training."""
    
    # Load raw images
    raw_images = tf.keras.preprocessing.image_dataset_from_directory(
        raw_dir,
        label_mode=None,
        image_size=image_size,
        batch_size=32
    )

    # Load reference images (ground truth)
    reference_images = tf.keras.preprocessing.image_dataset_from_directory(
        reference_dir,
        label_mode=None,
        image_size=image_size,
        batch_size=32
    )

    # Normalize both sets of images
    raw_images = raw_images.map(lambda x: (x / 127.5) - 1)  # Normalize to [-1, 1]
    reference_images = reference_images.map(lambda x: (x / 127.5) - 1)  # Normalize to [-1, 1]

    return tf.data.Dataset.zip((raw_images, reference_images))

