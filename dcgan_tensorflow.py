import os
import time

os.environ["TF_CPP_MIN_LOG_LRVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
# import numpy as np

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


batchSize = 64  # We set the size of the batch.
# imageSize = 64  # We set the size of the generated images (64x64).

dataset = keras.preprocessing.image_dataset_from_directory(
    directory='./data/celeb_dataset', label_mode=None, image_size=(64, 64), batch_size=batchSize, shuffle=True
).map(lambda x: x / 255.0)

discriminator = keras.Sequential(
    [
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.LeakyReLU(),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2D(256, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2D(512, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')

    ]
)

generator = keras.Sequential(
    [
        layers.Input(shape=(100,)),
        layers.Dense(8 * 8 * 100),
        layers.Reshape((8, 8, 100)),
        layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", use_bias=False, activation="tanh"),

    ]
)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# The discriminator and the generator optimizers are different since you will train two networks separately.
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 30
noise_dim = 100

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, noise):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        # Training the discriminator with a real and fake image of the dataset
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_weights)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_weights)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_weights))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_weights))

    return gen_loss, disc_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for i, image in enumerate(tqdm(dataset)):
            noise = tf.random.normal([batchSize, noise_dim])
            generated_images = generator(noise, training=True)
            training = train_step(image, noise)

            print('\n[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Time for epoch %d is %d sc' % (
            epoch, epochs, i, len(dataset), training[0], training[1], epoch + 1, time.time() - start))

            if i % 100 == 0:
                keras.preprocessing.image.array_to_img(image[0]).save('%s/real_samples.png' % "./results")
                keras.preprocessing.image.array_to_img(generated_images[0]).save(
                    '%s/fake_samples_epoch_%03d.png' % ("./results", epoch))


train(dataset, EPOCHS)
