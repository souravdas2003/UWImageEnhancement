import tensorflow as tf
from tensorflow.keras import layers

def generator_model():
    """Builds the Generator part of the GAN."""
    inputs = layers.Input(shape=[256, 256, 3])

    down_stack = [
        layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2D(128, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2D(256, (4, 4), strides=2, padding='same', activation='relu')
    ]

    up_stack = [
        layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same', activation='relu'),
        layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same', activation='tanh')
    ]

    x = inputs
    for down in down_stack:
        x = down(x)
    
    for up in up_stack:
        x = up(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator_model():
    """Builds the Discriminator part of the GAN."""
    inputs = layers.Input(shape=[256, 256, 3])
    x = layers.Conv2D(64, (4, 4), strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(1, (4, 4), padding='same')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


class FUNIE_GAN:
    def __init__(self):
        self.generator = generator_model()
        self.discriminator = discriminator_model()

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
