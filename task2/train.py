import os
from dataset import DatasetGenerator
from model import Facenet

CWD = os.path.dirname(__file__)

def train(epochs: int = 20, 
          batch_size: int = 64, 
          freeze_layers: int = 250,
          load_pretrained_weights: bool = True, 
          load_trained_weights: bool = False):
    
    if not os.path.exists(os.path.join(CWD, "weights")):
        load_trained_weights = False
    
    if load_pretrained_weights and load_trained_weights:
        load_pretrained_weights = False



    data_path = os.path.join(CWD, 'data/Extracted Faces')
    data_folder = os.listdir(data_path)
    data_generator = DatasetGenerator(data_path, data_folder, preprocess=True, batch_size=batch_size)
    model = Facenet(learning_rate = 0.001,
                    freeze_layers = freeze_layers,
                    pretrained_weights = load_pretrained_weights,
                    trained_weights = load_trained_weights
                    )
    model.fit(data_generator, epochs)



