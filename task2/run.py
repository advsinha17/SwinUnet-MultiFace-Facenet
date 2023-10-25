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
train_parser.add_argument('--load_pretrained_weights', action='store_true', 
                          help='Load pretrained Facenet weights.')
train_parser.add_argument('--no-load_pretrained_weights', dest='load_pretrained_weights', action='store_false',
                          help='Do not load pretrained Facenet weights.')
train_parser.add_argument('--load_trained_weights', action='store_true',
                          help='''Whether to load trained/finetuned weights (if exists) 
                            from previous trainings. If both --load_pretrained_weights 
                            and --load_trained_weights are True, then trained/finetuned weights will be loaded.
                            If model has not been trained or trained/finetuned weights are not available, pretrained weights will 
                            be loaded based on value of --load_pretrained_weights argument. Trained weights will only be available in the `weights` folder. False if not given.''')

train_parser.set_defaults(load_pretrained_weights=True, load_trained_weights=False)

# Inference arguments
inference_parser = subparsers.add_parser('inference')
inference_parser.add_argument('--input_folder', type=str, default="input_images/",
                                help='''Path to the folder containing images for inference. Directory must exist. Defaults to `input_images/`''')
inference_parser.add_argument('--output_folder', type=str, default="output_images/",
                                help='''Output folder path. Subdirectories will be created for each person
                                in the images. An existing directory with the same 
                                path will be overwritten. Defaults to `output_images/`''')
inference_parser.add_argument('--use_pretrained_weights', action='store_true', 
                          help='Load pretrained Facenet weights.')
inference_parser.add_argument('--no-use_pretrained_weights', dest='load_pretrained_weights', action='store_false',
                          help='Do not load pretrained Facenet weights.')
inference_parser.add_argument('--use_trained_weights', action='store_true',
                          help='''Whether to use trained/finetuned weights (if exists) 
                            from previous trainings. If both --load_pretrained_weights 
                            and --load_trained_weights are True, then trained/finetuned weights will be used.
                            If model has not been trained or trained/finetuned weights are not available, pretrained weights will 
                            be loaded. Trained weights will only be available in the `weights` folder. False if not given.''')

inference_parser.set_defaults(use_pretrained_weights=True, use_trained_weights=False)

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