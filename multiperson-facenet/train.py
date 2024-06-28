import os
from dataset import DatasetGenerator
from model import Facenet

CWD = os.path.dirname(__file__)

def train(epochs: int = 20, 
          batch_size: int = 64, 
          freeze_layers: int = 250,
          load_pretrained_weights: bool = True, 
          load_trained_weights: bool = False):
    """
    Train the FaceNet model on the given dataset.

    Parameters:
        epochs (int): Number of training epochs.
        batch_size (int): Size of the batches for training.
        freeze_layers (int): Number of initial layers to freeze.
        load_pretrained_weights (bool): If True, will load pretrained weights for the model.
        load_trained_weights (bool): If True, will load previously trained weights.

    """
    
    weights_path = os.path.join(CWD, "weights")

    if load_pretrained_weights and load_trained_weights:
        if os.path.exists(weights_path):
            print("Warning: Both load_pretrained_weights and load_trained_weights given True. Using trained weights.")
            load_pretrained_weights = False
        else:
            print("Warning: Both load_pretrained_weights and load_trained_weights given True but trained weights not available. Using pretrained weights.")
            load_trained_weights = False

    elif load_trained_weights:
        if not os.path.exists(weights_path):
            print("Warning: Trained weights not available. Weights will be randomly initialized.")
            load_trained_weights = False

    data_path = os.path.join(CWD, 'data/Extracted Faces')
    data_folder = os.listdir(data_path)
    images_folder = {}
    for folder in data_folder:
        path = os.path.join(data_path, folder)
        if os.path.isdir(path):
            images_folder[folder] = len(os.listdir(path))

    data_generator = DatasetGenerator(data_path, images_folder, preprocess=True, batch_size=batch_size)
    model = Facenet(learning_rate = 0.001,
                    freeze_layers = freeze_layers,
                    pretrained_weights = load_pretrained_weights,
                    trained_weights = load_trained_weights
                    )
    model.fit(data_generator, epochs)

    print("Training completed successfully.")

if __name__ == "__main__":
    train(epochs = 20, 
        batch_size = 64,
        freeze_layers = 250,
        load_pretrained_weights = True,
        load_trained_weights = False)