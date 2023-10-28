## Task 1

Implementation of the [Swin Transformer](https://arxiv.org/abs/2103.14030) and [ConvNeXT](https://arxiv.org/abs/2201.03545) models can be found `swin_model.py` and `convnext_model.py` respectively.

The Swin Transformer model uses a window based self-attention mechanism bringing the power of transformers to vision tasks. Earlier implementations of vision tasks with transformers computed global self-attention, where self-attention for all pixel pairs was calculated, making it computationally expensive. The Swin Transformer model, however, calculates self-attention within each non-overlapping window. To introduce cross-window connections, the swin transformer uses shifted windows between consecutive transformer blocks.

The ConvNeXT model implements a multi-stage design similar to te Swin Transformer. Here, however, each stage is made of entirely convolutional blocks. The results presented in the paper show that the ConvNeXT model competes with the Swin Transformer while maintaining the ease of implementation of ConvNets.

I have incorporated these models into the UNet architecture to perform image segmentation. Both models have been used as encoder blocks in the UNet architecture. These models can be found in `SwinUNet.py` and `ConvUNet.py`. The model graphs are present in the `plots/` folder.

### Results

The models have been trained with a few different hyperparameters. The training with the best results are presented in `train.ipynb` and the corresponding plots can be found in `plots/train_plots`. The results of training the models with some different hyperparameters are presented in `experiments`.

In `train_diff_decay.ipynb` the models are trained with a different weight decay and for a few extra epochs.
In `train_lr_scheduleripynb` the models are trained with a Cosine Decay learning rate scheduler.
In `train_smaller_net.ipynb` the models are trained with a different configuration of the Swin Transformer and ConvNeXT blocks, using fewer number of blocks per stage.

### Bonus Task

For the bonus task, I have implemented a disentangled VAE model. The disentangled VAE model aims to learn the factors of variation in the data in a disentangled or independent manner within the latent space.

I have trained the model on a subset of the [dSprites dataset](https://github.com/google-deepmind/dsprites-dataset/tree/master). I still need to experiment with some different learning rates and values of beta to find the optimal values for convergence of the reconstruction loss and achieve disentanglement.
