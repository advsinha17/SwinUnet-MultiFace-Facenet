# CRUx Round 3

### Task1: Swin UNET, checkout `task1/README.md`.
### Task2: Multi-face detection in facenet, checkout `task2/README.md`.


## Task 1: Swin & ConvNext Architechture implementation

1. Understanding of the paper Early Visual Concept Learning with Unsupervised Deep Learning
2. Understand, implement and compare the architectures of **Swin Transformer** and **ConvNeXT** on a non-vanilla classification task of your choosing.
   1. For clarification, a vanilla classification task is one where you have a labelled set of images on which you directly perform single or multi-class classification.
3. Bonus Task: Implementation of the Disentangled VAE (Task 1.1)

## Task 2: ML in production: Multi-face detection in facenet

1. Create folders for each individual in which images, wherein their faces are detected, are stored. (Multiple copies of images obviously now exist as multiple people are in the same photo.) However, there is one addition. You will need to write it in such a way that these folders can be created regardless of the input data (number of people in an image, size of image, etc.).
2. The final runnable should just have the user add photos to a folder (no required preprocessing) and run a file that will do everything necessary.
3. Ideally, no training should happen at inference time.

## Versions Used

```
python: 3.10
tensorflow: 2.13.0
keras: 2.13.1
```
