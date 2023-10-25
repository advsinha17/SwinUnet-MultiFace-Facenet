import tensorflow as tf
from swin_model import PatchAndEmbed, StageTransformerBlocks, PatchMerging
from UNetBlocks import UpsamplingBlock


class SwinUNet(tf.keras.Model):


    def __init__(self,  input_dim, n_classes = 3, depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24], window_size = 7, patch_size = 4, embed_dim = 96, attn_drop_rate = 0., proj_drop_rate = 0.):
         
        super(SwinUNet, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Downsampling blocks
        self.patch_embed = PatchAndEmbed(input_dim // patch_size, patch_size, embed_dim)
        self.downsampling_block1 = StageTransformerBlocks(depths[0], 
                                                        input_dim // patch_size, 
                                                        embed_dim, 
                                                        num_heads[0],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)

        self.merge1 = PatchMerging((input_dim // patch_size, input_dim // patch_size), embed_dim)
        self.downsampling_block2 = StageTransformerBlocks(depths[1], 
                                                        input_dim // (2 * patch_size), 
                                                        embed_dim * 2, 
                                                        num_heads[1],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)

        self.merge2 = PatchMerging((input_dim // (2 * patch_size), input_dim // (2 * patch_size)), 2 * embed_dim)
        self.downsampling_block3 = StageTransformerBlocks(depths[2], 
                                                        input_dim // (4 * patch_size), 
                                                        embed_dim * 4, 
                                                        num_heads[2],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)

        self.merge3 = PatchMerging((input_dim // (4 * patch_size), input_dim // (4 * patch_size)), 4 * embed_dim)
        self.downsampling_block4 =  StageTransformerBlocks(depths[3], 
                                                        input_dim // (8 * patch_size), 
                                                        embed_dim * 8,
                                                        num_heads[3],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.downsampling_block5 =  StageTransformerBlocks(depths[3], 
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



        # Downsampling blocks
        x = self.patch_embed(input_tensor)
        down1 = self.downsampling_block1(x, training = training)
        x = self.merge1(down1)
        down1= tf.reshape(down1, [-1, self.input_dim // self.patch_size, self.input_dim // self.patch_size, self.embed_dim])
        down2 = self.downsampling_block2(x, training = training)
        x = self.merge2(down2)
        down2 = tf.reshape(down2, [-1, self.input_dim // (2 * self.patch_size), self.input_dim // (2 * self.patch_size), 2 * self.embed_dim])
        down3 = self.downsampling_block3(x, training = training)
        x = self.merge3(down3)
        down3 = tf.reshape(down3, [-1, self.input_dim // (4 * self.patch_size), self.input_dim // (4 * self.patch_size), 4 * self.embed_dim])
        down4 = self.downsampling_block4(x, training = training)
        down5 = self.downsampling_block5(down4, training = training)
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
    