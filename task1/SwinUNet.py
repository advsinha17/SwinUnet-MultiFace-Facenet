import tensorflow as tf
from swin_model import PatchAndEmbed, StageTransformerBlocks, PatchMerging, SwinTransformer
from UNetBlocks import UpsamplingBlock


class SwinUNet(tf.keras.Model):
    """
    A SwinUNet architecture combining Swin Transformer blocks with a U-Net for segmentation tasks.

    Args:
        input_dim (int): Dimension of the input image.
        n_classes (int): Number of output classes for segmentation. Default is 3.
        depths (list of int): Number of blocks in each StageTransformerBlock. Default is [2, 2, 6, 2].
        num_heads (list of int): Number of attention heads in each StageTransformerBlock. Default is [3, 6, 12, 24].
        window_size (int): Window size for the Swin Transformer. Default is 7.
        patch_size(int): Number of pixels along one dimension of each patch. Default is 4.
        embed_dim(int): Embedding dimension for the patches. Default is 96.
        attn_drop(float): Dropout rate for the dropout layer after the attention mechanism. Default is 0.
        proj_drop(float): Dropout rate for the dropout layer after projection. Default is 0.
    """


    def __init__(self,  input_dim, n_classes = 3, depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24], window_size = 7, patch_size = 4, embed_dim = 96, attn_drop_rate = 0., proj_drop_rate = 0.):
         
        super(SwinUNet, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Downsampling blocks
        self.encoder = SwinTransformer(n_classes, input_dim, patch_size, embed_dim, depths, num_heads, window_size, attn_drop_rate, proj_drop_rate, False)
        self.last_transformer =  StageTransformerBlocks(depths[3], 
                                                        input_dim // (8 * patch_size), 
                                                        embed_dim * 8,
                                                        num_heads[3],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
    
        # Upsampling blocks
        self.upsampling_block1 = UpsamplingBlock(embed_dim * 4)
        self.upsampling_block2 = UpsamplingBlock(embed_dim * 2)
        self.upsampling_block3 = UpsamplingBlock(embed_dim)
        self.upsampling_block4 = UpsamplingBlock(3, last=True)

        # conv layer
        self.conv = tf.keras.layers.Conv2D(embed_dim,
                                           kernel_size = (3, 3),
                                           padding = 'same',
                                           kernel_initializer = 'he_normal')
        
        # BatchNorm layer
        self.bn = tf.keras.layers.BatchNormalization()

        # Output conv layer
        self.output_conv = tf.keras.layers.Conv2D(n_classes, 1, padding = 'same')

    def call(self, input_tensor, training = False):
        """ Forward pass for SwinUNet."""
        # Downsampling blocks
        x = self.encoder.patch_embed(input_tensor)
        down1 = self.encoder.transformers_stage1(x)
        x = self.encoder.merge1(down1)
        down1 = tf.reshape(down1, [-1, self.input_dim // self.patch_size, self.input_dim // self.patch_size, self.embed_dim])
        down2 = self.encoder.transformers_stage2(x)
        x = self.encoder.merge2(down2)
        down2 = tf.reshape(down2, [-1, self.input_dim // (2 * self.patch_size), self.input_dim // (2 * self.patch_size), 2 * self.embed_dim])
        down3 = self.encoder.transformers_stage3(x)
        x = self.encoder.merge3(down3)
        down3 = tf.reshape(down3, [-1, self.input_dim // (4 * self.patch_size), self.input_dim // (4 * self.patch_size), 4 * self.embed_dim])
        down4 = self.encoder.transformers_stage4(x)
        down5 = self.last_transformer(down4, training = training)
        down4 = tf.reshape(down4, [-1, self.input_dim // (8 * self.patch_size), self.input_dim // (8 * self.patch_size), 8 * self.embed_dim])
        down5 = tf.reshape(down5, [-1, self.input_dim // (8 * self.patch_size), self.input_dim // (8 * self.patch_size), 8 * self.embed_dim])

        # Upsampling blocks
        up1 = self.upsampling_block1(down5, down3, training = training) 
        up2 = self.upsampling_block2(up1, down2, training = training)
        up3 = self.upsampling_block3(up2, down1, training = training)
        up4 = self.upsampling_block4(up3, input_tensor, training = training)

        # conv layer
        x = self.conv(up4)

        # BatchNorm layer
        x = self.bn(x, training = training)

        # ReLU layer
        x = tf.nn.relu(x)

        # Ouput conv layer
        outputs = self.output_conv(x)

        return outputs
    
    def build_graph(self, input_shape):

        x = tf.keras.layers.Input(shape = input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))
    