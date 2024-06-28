## Task 2

For task 2, I have created an end-to-end multi-face detection system.

### Features

- **Automated Face Detection**: Process a directory of images and detect individual faces.
- **Individual Folder Creation**: For each identified individual, the system will create a separate folder and store all the images containing their face. This means that if an image has multiple faces, it will appear in multiple folders.
- **Adaptable to Different Input Data**: Whether you have images with one or multiple faces, regardless of the image size, the system can handle it.
- **Training and Inference Phases**: Can train the model on provided data and then use it for inference on any dataset you have.

### How to Use

For full details on using the face detection system, check `run.py`.

Data to train the model can be downloaded [here](https://drive.google.com/drive/folders/1tlUKKANRPrbzNmSln-hAxKw6-ddiHs9t?usp=sharing).

Fine tuned weights for this dataset can be downloaded [here](https://drive.google.com/drive/folders/1LmhIFNG_Pz3tMmYrwq9L-FPw5gdGg1ZF?usp=sharing).

#### 1. Training

- To train the model on provided data, run the following command:

  ```
  python run.py train --epochs <number_of_epochs> --batch_size <batch_size>
  ```

- Optional arguments for training can also be included, such as --freeze_layers, --load_pretrained_weights, --no-load_pretrained_weights, and --load_trained_weights.

### 2. Inference

- To run inference, place all the images you wish to process in a folder (no preprocessing required).

- Run the inference command:

```
python run.py inference --input_folder <path_to_input_folder> --output_folder <path_to_output_folder>
```

- Optional arguments for inference include --use_pretrained_weights and --use_trained_weights.
