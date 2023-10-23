import cv2
import os
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from model import Facenet
import numpy as np
from sklearn.cluster import DBSCAN
import shutil
from typing import List, Optional, Union

CWD = os.path.dirname(__file__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMAGE_SIZE = (160, 160)

def detect_faces(detector: MTCNN, 
                 image_path: str) -> Optional[List[np.ndarray]]:

    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)
    list_faces = []
    if faces:
        for face in faces:
            x, y, width, height = face['box']
            face_image = img[y:y+height, x:x+width]
            face_image = cv2.resize(face_image, IMAGE_SIZE)
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            list_faces.append(face_image)

        return list_faces

    return None

def get_embedding(model: Facenet, 
                  face_image: np.ndarray) -> np.ndarray:
    face_image = (face_image - 127.5) / 128.
    face_image = np.reshape(face_image, (1,) + IMAGE_SIZE + (3,))
    embedding = model.predict(face_image)
    return embedding

def cluster_faces(embeddings: np.ndarray, 
                  image_paths: List[str], 
                  output_dir: str = 'output_images/'):
    clusterer = DBSCAN(eps = 10, min_samples = 2, metric = "euclidean", n_jobs = -1)
    clusterer.fit(embeddings)
    labels_id = clusterer.labels_

    output_path = os.path.join(CWD, output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, label in enumerate(labels_id):
        cluster_folder = os.path.join(output_path, f'person_{label + 1}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        shutil.copy(image_paths[idx], os.path.join(cluster_folder, os.path.basename(image_paths[idx])))


def detect(input_dir: str = 'input_images/', 
            output_dir: str = 'output_images/',
            use_pretrained_weights: bool = True,
            use_trained_weights: bool = False):
    
    if not os.path.exists(output_dir):
        use_pretrained_weights = True
        use_trained_weights = False

    
    if use_pretrained_weights and use_trained_weights:
        use_pretrained_weights = False
    
    detector = MTCNN()
    model = Facenet()
    input_dir_path = input_dir
    if not os.path.exists(input_dir_path):
        input_dir_path = os.path.join(CWD, 'input_dir')
    if not os.path.exists(input_dir_path):
        raise FileNotFoundError('Input directory does not exist')
    images = os.listdir(input_dir_path)
    all_embeddings = []
    all_face_image_paths = []
    for image in images:
        image_path = os.path.join(input_dir, image)
        face_list = detect_faces(detector, image_path)

        if face_list:
            for face in face_list:
                embedding = get_embedding(model, face)
                all_embeddings.append(embedding)
                all_face_image_paths.append(image_path)
    cluster_faces(np.squeeze(all_embeddings), all_face_image_paths, output_dir)
    

        
