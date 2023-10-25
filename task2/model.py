import numpy as np
from architecture import InceptionResNetV1
from tqdm import tqdm
import os
import tensorflow as tf

CWD = os.path.dirname(__file__)

class Facenet:

    """
    Class that implements the Facenet model for face recognition.

    The Facenet architecture is defined in the architecture.py file. This class provides a high-level interface
    for loading, fine-tuning, and using the Facenet model.

    """

    def __init__(self, learning_rate = 1e-3, freeze_layers = 250, pretrained_weights = True, trained_weights = False):

        """
        Initialize a Facenet instance.

        Args:
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
            freeze_layers (int, optional): Number of layers of the pre-trained model to freeze. Defaults to 250.
            default_weights (bool, optional): Whether to use the default weights or fine-tuned weights. Defaults to True.
        """
        super(Facenet, self).__init__()

        # Create the base InceptionResNetV1 model
        self.model = InceptionResNetV1()

        # Use default or fine-tuned weights
        self.pretrained_weights = pretrained_weights
        self.trained_weights = trained_weights

        # Initialize the optimizer with the specified learning rate
        self.optimizer = tf.optimizers.legacy.Adam(learning_rate=learning_rate)

         # Load pre-trained weights
        if self.pretrained_weights:
            self.model.load_weights(os.path.join(CWD, "model_data/facenet_keras_weights.h5"))
        elif self.trained_weights:
            self.model.load_weights(os.path.join(CWD, "weights/model-final.h5"))

        # Set layers as trainable based on freeze_layers
        trainable = False
        i = 0
        for layer in self.model.layers:
            i += 1
            if i == freeze_layers:
                trainable = True
            layer.trainable = trainable

    @tf.function
    def _train_step(self, anchor_batch, positive_batch, negative_batch):
        """
        Perform a single training step for the Facenet model using triplet loss.

        Args:
            anchor_batch (tf.Tensor): Batch of anchor images.
            positive_batch (tf.Tensor): Batch of positive images.
            negative_batch (tf.Tensor): Batch of negative images.

        Returns:
            loss (float): The computed triplet loss for the current batch.
        """

        with tf.GradientTape() as tape:
            anchor_embeddings = self.model(anchor_batch)
            positive_embeddings = self.model(positive_batch)
            negative_embeddings = self.model(negative_batch)
            loss = self._triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    
    def _triplet_loss(self, anchor_embedding, positive_embedding, negative_embedding, margin=0.2):
        """
        Calculate the triplet loss for a batch of anchor, positive, and negative embeddings.

        Args:
            anchor_embedding (tf.Tensor): Embeddings of anchor images.
            positive_embedding (tf.Tensor): Embeddings of positive images.
            negative_embedding (tf.Tensor): Embeddings of negative images.
            margin (float, optional): Margin parameter for triplet loss. Defaults to 0.2.

        Returns:
            loss (float): The computed triplet loss.
        """
        pos_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis=-1)
        neg_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis=-1)
        
        loss = tf.maximum(pos_distance - neg_distance + margin, 0.0)
        loss = tf.reduce_mean(loss)
        
        return loss

    def fit(self, inputs, epochs):
        """
        Fine-tune the Facenet model using triplet loss with given input triplets.

        Args:
            inputs (tf.data.Dataset): A dataset of triplets (anchor, positive, negative).
            epochs (int): The number of training epochs.

        Returns:
            history (dict): A dictionary containing training history (e.g., 'train_loss').
        """
        train_loss = []
        os.makedirs(os.path.join(CWD, 'weights'), exist_ok=True)
        for epoch in range(1, epochs + 1):
            epoch_loss = []
            with tqdm(inputs, unit="batch") as tepoch:
                for data in tepoch:
                    anchor_batch, positive_batch, negative_batch = data
                    loss = self._train_step(anchor_batch, positive_batch, negative_batch)
                    epoch_loss.append(loss)
            total_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            train_loss.append(total_epoch_loss)

            print(f"Epoch {epoch}: Loss on train = {total_epoch_loss:.5f}")
            if epoch % 5 == 0:
                self.model.save_weights(os.path.join(CWD, 'weights/model.h5'))

        self.model.save_weights(os.path.join(CWD, os.path.join('weights/model-final.h5')))
        history = {
            'train_loss': train_loss,
        }
        return history

    def predict(self, input, verbose = 1):
        """
        Make predictions using the Facenet model.

        Args:
            input (tf.Tensor): Input data for making predictions.
            verbose (int): verbose value for model.predict
        Returns:
            tf.Tensor: Predicted embeddings.
        """
        return self.model.predict(input, verbose = verbose)
    
    def summary(self):
        """
        Display a summary of the Facenet model's architecture.

        Returns:
            str: Model summary.
        """
        return self.model.summary()


if __name__ == "__main__":
    model = Facenet()
    model.summary()

