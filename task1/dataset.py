import tensorflow as tf
import os
import numpy as np
import random
import cv2
import xml.etree.ElementTree as ET

CWD = os.path.dirname(__file__)
classes = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 
            'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
id_to_label = {k: v for k, v in enumerate(classes)}
label_to_id = {v: k for k, v in id_to_label.items()}


def extract_single_xml_file(filename):
    parse = ET.parse(filename)
    objects = parse.findall('./object')
    fname = parse.find('./filename').text
    dicts = [{obj.find('name').text: [int(float(obj.find('bndbox/xmin').text)),
                                      int(float(obj.find('bndbox/ymin').text)),
                                      int(float(obj.find('bndbox/xmax').text)),
                                      int(float(obj.find('bndbox/ymax').text))]}
                                      for obj in objects]
    return {'filename': fname, 'objects': dicts}

def get_annotations(annot_dir):
    annotations = []
    for file in sorted(os.listdir(annot_dir)):
        annotation = extract_single_xml_file(os.path.join(annot_dir, file))
        image_objects = []
        for object in annotation['objects']:
            image_objects.append(object)

        if len(image_objects) == 1:
            annotation['class'] = list(image_objects[0].keys())[0]
            annotation['bbox'] = list(image_objects[0].values())[0]
            annotation.pop('objects')
            annotations.append(annotation)

    return annotations

def split_data(annotations, train_split = 0.8):
    random.shuffle(annotations)
    train_annot = annotations[:int(len(annotations) * train_split)]
    val_annot = annotations[int(len(annotations) * train_split):]
    return train_annot, val_annot

class DataGenerator(tf.keras.utils.Sequence):

    '''
    Dataset used is PASCAL VOS 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).
    Generates batches of images where each image contains only 1 object.

    Args:
        annotations: List annotation dictionaries.
        image_size: Size of the image. Defaults to (224, 224).
        batch_size: Size of the batch. Defaults to 64.
    '''

    def __init__(self, annotations, image_size = (224, 224), batch_size = 64):
        super(DataGenerator, self).__init__()
        self.batch_size = batch_size
        self.annotations = annotations
        self.list_images = [i['filename'] for i in self.annotations]
        self.images_dict = {}
        for i in self.annotations:
            self.images_dict[i['filename']] = i['bbox'].copy()
            self.images_dict[i['filename']].insert(0, label_to_id[i['class']])


        self.image_size = image_size

    def __len__(self):

        return int(np.ceil(len(self.list_images) / self.batch_size))
    
    def read_image(self, filename):
        path = os.path.join(CWD, 'data/JPEGImages', filename)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = (image - 127.5) / 128.
        return image

    def __getitem__(self, index):
        batch = self.list_images[index * self.batch_size: (index + 1) * self.batch_size]
        images = []
        labels = []
        for image in batch:
            img = self.read_image(image)
            images.append(img)
            label = [self.images_dict[image]]
            labels.append(label)

        return images, labels
    

if __name__ == '__main__':
    print(label_to_id)
    print(id_to_label)
    annots = get_annotations(os.path.join(CWD, 'data/Annotations'))
            
