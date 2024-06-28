import os
import random
import numpy as np
import cv2
os.environ["TF_CPP_MIN_LOG"] = "2"

import tensorflow as tf
CWD = os.path.dirname(__file__)
IMAGE_SIZE = (160, 160)

class DatasetGenerator(tf.keras.utils.Sequence):

    """
    Class that generates batches of (anchor, positive, negative) triplets after preprocessing the images.
    """

    def __init__(self, images_dir, folder_list, preprocess = True, batch_size = 128):
        """
        Initialize a DatasetGenerator instance.

        Args:
            images_dir (str): Directory containing the images.
            folder_list (dict): A dictionary where keys are folder names and values are the number of files in each folder.
            preprocess (bool, optional): Whether to preprocess the images. Defaults to True.
            batch_size (int, optional): Size of the batch generated. Defaults to 128.
        """
        super(DatasetGenerator, self).__init__()
        self.batch_size = batch_size
        self.images_dir = images_dir
        self.folder_list = folder_list
        self.preprocess = preprocess
        self.triplets = self._create_triplets()

    def _read_image(self, filename):

        """
        Reads an image from the specified filename.

        Args:
            filename (tuple): Tuple containing folder name and image filename.

        Returns:
            image (np.ndarray): NumPy array representing the image.
        """
        path = os.path.join(CWD, "data/Extracted Faces", filename[0], filename[1]) 
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        if self.preprocess:
            image = (image - 127.5)/128.0
        return image

    
    def _create_triplets(self, max_files = 10):

        """
        Creates (anchor, positive, negative) triplets from the given folders.

        Args:
            max_files (int, optional): Maximum number of files to use from a given class. Defaults to 10.

        Returns:
            triplets (list): List of (anchor, positive, negative) triplets.
        """
        triplets = []
        folders = list(self.folder_list.keys())

        for folder in folders:
            path = os.path.join(self.images_dir, folder)
            images = list(os.listdir(path))[:max_files]
            num_imgs = len(images)
            if num_imgs == 1:
                continue

            for i in range(num_imgs - 1):
                for j in range(i + 1, num_imgs):
                    anchor = (folder, images[i])
                    positive = (folder, images[j])
                    neg_folder = folder
                    while neg_folder == folder:
                        neg_folder = random.choice(folders)
                    neg_file_index = random.randint(0, self.folder_list[neg_folder] - 1)
                    neg_file = list(os.listdir(os.path.join(CWD, 'data/Extracted Faces', neg_folder)))[neg_file_index]
                    negative = (neg_folder, neg_file)

                    triplets.append((anchor, positive, negative))

        random.shuffle(triplets)
        return triplets

    
    def __len__(self):

        """
        Returns the number of batches.

        Returns:
            int: Number of batches.
        """

        return int(np.ceil(len(self.triplets) / self.batch_size))
    
    def __getitem__(self, index):

        """
        Generates a batch of data.

        Args:
            index (int): Index of the batch to generate.

        Returns:
            tuple: A tuple containing three NumPy arrays (anchors, positives, negatives).
        """
        batches = self.triplets[index * self.batch_size:(index + 1) * self.batch_size]
        anchors = []
        positives = []
        negatives = []
        for triplet in batches:
            anchor, positive, negative = triplet
            anchors.append(self._read_image(anchor))
            positives.append(self._read_image(positive))
            negatives.append(self._read_image(negative))

        anchors = np.array(anchors)
        positives = np.array(positives)
        negatives = np.array(negatives)


        return ([anchors, positives, negatives])

