import tensorflow as tf
import numpy as np
from model import BetaVAE
from dataset import DataGenerator
from tqdm import tqdm


def train(model, epochs, data, optimizer = tf.keras.optimizers.Adam()):

    model.compile(optimizer=optimizer)

    for epoch in range(epochs):
        progress_bar = tqdm(data, total=len(data), unit="batch")
        for batch in progress_bar:
            losses  = model.train_step(batch)
            progress_bar.set_description(f"Loss: {losses['loss'].numpy():.4f}, Reconstruction Loss: {losses['reconstruction_loss'].numpy():.4f}, kl_div_loss: {losses['kl_loss'].numpy():.4f}")
        print(f"End of Epoch {epoch + 1}, Loss: {losses['loss'].numpy():.4f}, Reconstruction Loss: {losses['reconstruction_loss'].numpy():.4f}, kl_div_loss: {losses['kl_loss'].numpy():.4f}")



if __name__ == "__main__":
    data_generator = DataGenerator()
    epochs = 20
    model = BetaVAE(10, out_channels=1, beta=4.0)
    train(model, epochs, data_generator)