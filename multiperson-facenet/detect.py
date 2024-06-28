import cv2
import os
from mtcnn import MTCNN
from model import Facenet
import numpy as np
from sklearn.cluster import DBSCAN
import shutil
from typing import List, Optional

CWD = os.path.dirname(__file__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMAGE_SIZE = (160, 160)

def detect_faces(detector: MTCNN, 
                 image_path: str) -> Optional[List[np.ndarray]]:
    """
    Detect faces in the given image using MTCNN.

    Args:
        detector (MTCNN): MTCNN face detection model.
        image_path (str): Path to the image.

    Returns:
        list_faces (list of np.ndarray): List of detected face images or None.
    """

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

def get_embedding(model: Facenet, face_images: List[np.ndarray]) -> np.ndarray:
    """
    Get embeddings for the detected face images using the Facenet model.

    Args:
        model (Facenet): Facenet model for generating embeddings.
        face_images (List[np.ndarray]): List of face images.

    Returns:
        embeddings (np.ndarray): Array of embeddings for the given face images.
    """
    face_images = [(face_image - 127.5) / 128. for face_image in face_images]
    face_images = np.stack(face_images, axis=0)
    embeddings = model.predict(face_images)
    return embeddings

def cluster_faces(embeddings: np.ndarray, 
                  image_paths: List[str], 
                  output_dir: str = 'output_images/'): 
    """
    Cluster detected face images based on their embeddings using DBSCAN.

    Args:
        embeddings (np.ndarray): Array of embeddings for the detected faces.
        image_paths (List[str]): List of paths to the original images.
        output_dir (str, optional): Directory to save clustered images. Defaults to 'output_images/'.

    The function saves the clustered images in the specified directory.
    """
    clusterer = DBSCAN(eps = 11.4, min_samples = 5, metric = "euclidean", n_jobs = -1)
    clusterer.fit(embeddings)
    labels_id = clusterer.labels_

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    output_path = os.path.join(output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, label in enumerate(labels_id):
        if label == -1:
            continue
        cluster_folder = os.path.join(output_path, f'person_{label + 1}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
        shutil.copy(image_paths[idx], os.path.join(cluster_folder, os.path.basename(image_paths[idx])))


def detect(input_dir: str = 'input_images/', 
            output_dir: str = 'output_images/',
            use_pretrained_weights: bool = True,
            use_trained_weights: bool = False):
    """
    Main function to detect and cluster faces.

    Args:
        input_dir (str, optional): Directory containing input images. Defaults to 'input_images/'.
        output_dir (str, optional): Directory to save clustered images. Defaults to 'output_images/'.
        use_pretrained_weights (bool, optional): Flag to use pretrained weights for the Facenet model. Defaults to True.
        use_trained_weights (bool, optional): Flag to use trained weights for the Facenet model. Defaults to False.

    The function prints the result status.
    """
    
    if not os.path.exists(input_dir):
        print("Error: Input directory does not exist!")
        return
    
    if not os.path.exists(os.path.join(CWD, 'weights')) and use_trained_weights:
        print("Error: Trained weights not available. Will use pretrained weights.")
        use_pretrained_weights = True
        use_trained_weights = False

    
    elif use_pretrained_weights and use_trained_weights:
        print("Warning:  Both trained and pretrained weights cannot be used. Will use trained wieghts.")
        use_pretrained_weights = False

    elif not use_pretrained_weights and not use_trained_weights:
        print("Warning: Must use either pretrained or trained weights. Will use pretrained weights.")
        use_pretrained_weights = True

    
    detector = MTCNN()
    model = Facenet()
    images = os.listdir(input_dir)
    all_embeddings = []
    all_face_image_paths = []
    all_face_images = []
    for image in images:
        image_path = os.path.join(input_dir, image)
        face_list = detect_faces(detector, image_path)

        if face_list:
            all_face_images.extend(face_list)
            all_face_image_paths.extend([image_path] * len(face_list))

    embeddings_batch = get_embedding(model, all_face_images)
    all_embeddings.extend(embeddings_batch)
    cluster_faces(np.squeeze(all_embeddings), all_face_image_paths, output_dir)

    print(f"Face detection completed. Check the results in the directory: {output_dir}")
    

if __name__ == "__main__":
    detect(input_dir = 'input_images', 
        output_dir = 'output_images',
        use_pretrained_weights = True,
        use_trained_weights = False)