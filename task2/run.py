import argparse
from train import train
from detect import detect

parser = argparse.ArgumentParser(description="Face Recognition System")

subparsers = parser.add_subparsers(dest="mode")

# Training arguments
train_parser = subparsers.add_parser('train')
train_parser.add_argument('--epochs', type=int, default=20,
                            help='Number of epochs to train. Defaults to 20.')
train_parser.add_argument('--batch_size', type=int, default=64,
                            help='Number of training images per batch. Defaults to 64.')
train_parser.add_argument('--freeze_layers', type=int, default=250,
                            help='Number of layers of Facenet model to freeze. Defaults to 250.')
train_parser.add_argument('--load_pretrained_weights', type=bool, default=True,
                            help='''Whether to load pretrained Facenet weights. If both load_pretrained_weights 
                            and load_finetuned_weights are True, then finetuned weights will be loaded. Defaults to True.''')
train_parser.add_argument('--load_trained_weights', type=bool, default=False,
                            help='''Whether to load trained/finetuned weights (if exists) 
                            from previous trainings. If both load_pretrained_weights 
                            and load_trained_weights are True, then trained/finetuned weights will be loaded.
                            If model has not been trained or trained/finetuned weights are not available, pretrained weights will 
                            be loaded based on value of --load_pretrained_weights argument. Defaults to False''')


# Inference arguments
inference_parser = subparsers.add_parser('inference')
inference_parser.add_argument('--input_folder', type=str, default="input_images/",
                                help='''Path to the folder containing images for inference. Requires
                                absolute path or path relative to `run.py` script. Raises FileNotFoundError if directory does
                                not exist. Defaults to `input_images/`''')
inference_parser.add_argument('--output_folder', type=str, default="output_images/",
                                help='''Output folder path. Subdirectories will be created for each person
                                in the images. Requires absolute path or path relative to `run.py` script. An existing directory with the same 
                                path will be overwritten. Defaults to `output_images/`''')
inference_parser.add_argument('--use_pretrained_weights', type=bool, default=True,
                            help='''Whether to use pretrained Facenet weights. If both use_pretrained_weights 
                            and use_finetuned_weights are True, then finetuned weights will be used. Defaults to True.''')
inference_parser.add_argument('--use_trained_weights', type=bool, default=False,
                            help='''Whether to use trained/finetuned weights (if exists) 
                            from training the model. If both use_pretrained_weights 
                            and use_trained_weights are True, then trained/finetuned weights will be used.
                            If model has not been trained or trained/finetuned weights are not available, pretrained weights will 
                            be used. Defaults to False''')

args = parser.parse_args()

if args.mode == 'train':
    train(epochs = args.epochs,
          batch_size = args.batch_size,
          freeze_layers = args.freeze_layers,
          load_pretrained_weights = args.load_pretrained_weights,
          load_trained_weights = args.load_trained_weights)
    
elif args.mode == 'inference':
    detect(input_dir = args.input_folder, 
           output_dir = args.output_folder,
           use_pretrained_weights = args.use_pretrained_weights,
           use_trained_weights = args.use_trained_weights)