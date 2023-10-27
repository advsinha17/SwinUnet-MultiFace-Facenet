import tensorflow as tf
from convnext_model import ConvNeXTBlock
from UNetBlocks import UpsamplingBlock


class ConvNeXTUNet(tf.keras.Model):


    def __init__(self,  input_dim, embed_dim, n_classes = 3, depths = [3, 3, 9, 3]):
        """
        A ConvNeXTUNet architecture combining ConvNeXT blocks with a U-Net for segmentation tasks.

        Args:
            input_dim (int): The spatial dimension of the input data.
            embed_dim (int): Initial embedding dimension for the network.
            n_classes (int, optional): Number of output classes for segmentation. Defaults to 3.
            depths (list, optional): List indicating the number of ConvNeXT blocks in each stage. Defaults to [3, 3, 9, 3].
        """
         
        super(ConvNeXTUNet, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Downsampling blocks
        self.stage1 = [
            tf.keras.layers.Conv2D(filters = embed_dim, kernel_size = 4, strides = 4),
            tf.keras.layers.LayerNormalization()
        ]
        self.downsampling_block1 = [ConvNeXTBlock(embed_dim) for _ in range(depths[0])]

        self.stage2 = [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 2, kernel_size = 2, strides = 2)] 
        self.downsampling_block2 = [ConvNeXTBlock(embed_dim * 2) for _ in range(depths[1])]

        self.stage3 = [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 4, kernel_size = 2, strides = 2)]
        self.downsampling_block3 = [ConvNeXTBlock(embed_dim * 4) for _ in range(depths[2])]

        self.stage4 = [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 8, kernel_size = 2, strides = 2)]
        self.downsampling_block4 = [ConvNeXTBlock(embed_dim * 8) for _ in range(depths[3])]
        self.downsampling_block5 = [ConvNeXTBlock(embed_dim * 8) for _ in range(depths[3])]

    
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

    def call(self, x, training = False):
        """Forward pass for the ConvNeXTUNet model."""

        input_tensor = x
        # Downsampling blocks
        for layer in self.stage1:
            x = layer(x)
        for layer in self.downsampling_block1:
            x = layer(x, training = training)
        down1 = x
        for layer in self.stage2:
            x = layer(x, training = training)
        down1= tf.reshape(down1, [-1, self.input_dim // 4, self.input_dim // 4, self.embed_dim])
        for layer in self.downsampling_block2:
            x = layer(x, training = training)
        down2 = x
        for layer in self.stage3:
            x = layer(x)
        down2 = tf.reshape(down2, [-1, self.input_dim // 8, self.input_dim // 8, 2 * self.embed_dim])
        for layer in self.downsampling_block3:
            x = layer(x, training = training)
        down3 = x
        for layer in self.stage4:
            x = layer(x)
        down3 = tf.reshape(down3, [-1, self.input_dim // 16, self.input_dim // 16, 4 * self.embed_dim])
        for layer in self.downsampling_block4:
            x = layer(x, training = training)
        down4 = x
        for layer in self.downsampling_block5:
            x = layer(x, training = training)
        down5 = x
        down4 = tf.reshape(down4, [-1, self.input_dim // 32, self.input_dim // 32, 8 * self.embed_dim])
        down5 = tf.reshape(down5, [-1, self.input_dim // 32, self.input_dim // 32, 8 * self.embed_dim])



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
    

