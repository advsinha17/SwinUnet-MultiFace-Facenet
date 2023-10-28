import tensorflow as tf
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class SampleLayer(tf.keras.layers.Layer):
    """Sample from a distribution with given mean and variance."""
    def __init__(self):
        super(SampleLayer, self).__init__()

    def call(self, inputs):
        mean, var = inputs
        eps = tf.random.normal(mean.get_shape().as_list())
        std = tf.math.exp(0.5 * var)
        return mean + std * eps


class Encoder(tf.keras.layers.Layer):
    """
    Encoder part of the VAE model.
    
    Args:
        latent_dim(int): Dimesion of the latent space.
    """

    def __init__(self, latent_dim):

        super(Encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.Flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(512, activation = 'relu')
        self.mean_layer = tf.keras.layers.Dense(latent_dim)
        self.var_layer = tf.keras.layers.Dense(latent_dim)
        self.sample_layer = SampleLayer()

    def call(self, input, training = None):
        x = self.conv1(input)
        x = self.bn1(x, training = training)
        x = self.conv2(x)
        x = self.bn2(x, training = training)
        x = self.conv3(x)
        x = self.bn3(x, training = training)
        x = self.conv4(x)
        x = self.bn4(x, training = training)
        x = self.Flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        sample = self.sample_layer([mean, var])

        return [mean, var, sample]
    
class Decoder(tf.keras.layers.Layer):

    """
    Decoder part of the VAE model.
    
    Args:
        out_channels(int): Number of channels in the reconstructed output. Default is 3.
    """

    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.reshape = tf.keras.layers.Reshape((4, 4, 32))
        self.conv_transpose1 = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv_transpose2 = tf.keras.layers.Conv2DTranspose(filters = 32, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv_transpose3 = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv_transpose4 = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = 'same', activation = 'relu')
        self.bn4 = tf.keras.layers.BatchNormalization()
        
        self.output_conv = tf.keras.layers.Conv2DTranspose(filters = out_channels, kernel_size = 4, strides = 1, padding = 'same', activation = 'sigmoid')

    def call(self, input, training = None):
        x = self.dense1(input)
        x = self.dense2(x)
        x = self.reshape(x)
        x = self.conv_transpose1(x)
        x = self.bn1(x, training = training)
        x = self.conv_transpose2(x)
        x = self.bn2(x, training = training)
        x = self.conv_transpose3(x)
        x = self.bn3(x, training = training)
        x = self.conv_transpose4(x)
        x = self.bn4(x, training = training)
        x = self.output_conv(x)
        return x

    
class BetaVAE(tf.keras.Model):
    
    """
    Beta Variational Autoencoder (VAE) model.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        beta (float, optional): Weighting factor for the KL divergence loss. Default is 4.0.
        out_channels (int, optional): Number of channels in the reconstructed output. Default is 3.
    """

    def __init__(self, latent_dim, beta = 4.0, out_channels = 3):

        super(BetaVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(out_channels)
        self.total_loss = tf.keras.metrics.Mean()
        self.recon_loss = tf.keras.metrics.Mean()
        self.kl_div_loss = tf.keras.metrics.Mean()
        self.beta = beta

    def call(self, input): 
        mean, var, sample = self.encoder(input)
        reconstruction = self.decoder(sample)

        return mean, var, reconstruction
    
    @property
    def metrics(self):
        return [
            self.total_loss,
            self.recon_loss,
            self.kl_div_loss,
        ]
    
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, var, sample = self.encoder(data)
            reconstruction = self.decoder(sample)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + var - tf.square(mean) - tf.exp(var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss.update_state(total_loss)
        self.recon_loss.update_state(recon_loss)
        self.kl_div_loss.update_state(kl_loss)
        return {
            "loss": self.total_loss.result(),
            "reconstruction_loss": self.recon_loss.result(),
            "kl_loss": self.kl_div_loss.result(),
        }



if __name__ == "__main__":

    input_tensor = tf.random.uniform((1, 64, 64, 1))
    model = BetaVAE(10)
    out_tensor = model(input_tensor)
    print(out_tensor[2].shape)
