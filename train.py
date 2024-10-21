from utils.utils import load_data
import tensorflow as tf
from models.funie_gan import FUNIE_GAN

def train_funie_gan():
    # Load raw and reference images
    dataset = load_data('data/raw/', 'data/reference/', image_size=(256, 256))

    # Initialize and compile the FUNIE-GAN model
    funie_gan = FUNIE_GAN()
    funie_gan.compile(
        g_optimizer=tf.keras.optimizers.Adam(2e-4),
        d_optimizer=tf.keras.optimizers.Adam(2e-4),
        loss_fn=tf.keras.losses.BinaryCrossentropy()
    )

    # Training Loop
    for epoch in range(100):
        for step, (raw_images, reference_images) in enumerate(dataset):
            # Add training logic here (omitted for brevity)

            # Save checkpoints periodically
            if step % 100 == 0:
                funie_gan.generator.save_weights('checkpoints/generator.h5')
                funie_gan.discriminator.save_weights('checkpoints/discriminator.h5')

if __name__ == "__main__":
    train_funie_gan()
