import tensorflow as tf
import os
from swin_utils import StochasticDepth
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class ConvNeXTBlock(tf.keras.layers.Layer):

    """
    Represents the basic building block of the ConvNeXT network.

    Args:
        embed_dim (int): Embedding dimension of the block.
    """

    def __init__(self, embed_dim):

        super(ConvNeXTBlock, self).__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size = 7, padding = "same")
        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.pointwise_conv1 = tf.keras.layers.Conv2D(filters = 4 * embed_dim, kernel_size = 1)
        self.gelu = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.pointwise_conv2 = tf.keras.layers.Conv2D(filters = embed_dim, kernel_size = 1)
        self.gamma = tf.Variable(1e-6 * tf.ones(embed_dim,))
        self.drop_path = StochasticDepth(0.1)

    def call(self, input, training = None):
        """
        Forward pass of the ConvNeXT Block.
        """
        x = self.depthwise_conv(input)
        x = self.norm_layer(x)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        x *= self.gamma
        x = self.drop_path(x, training = training) + input

        return x

class ConvNeXT(tf.keras.Model):

    """
    Represents the ConvNeXT network architecture.

    Args:
        num_classes (int): Number of output classes for classification.
        embed_dim (int): Embedding dimension of the first ConvNeXT block, default is 96.
        depths (list of ints): Number of ConvNeXT blocks in each stage, default is [3, 3, 9, 3].
    """

    def __init__(self, num_classes, embed_dim = 96, depths = [3, 3, 9, 3], include_top = True):
        
        super(ConvNeXT, self).__init__()

        self.stage1 = [
            tf.keras.layers.Conv2D(filters = embed_dim, kernel_size = 4, strides = 4),
            tf.keras.layers.LayerNormalization()
        ]
        self.stage1_block = [ConvNeXTBlock(embed_dim) for _ in range(depths[0])]

        self.stage2 = [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 2, kernel_size = 2, strides = 2)] 
        self.stage2_block = [ConvNeXTBlock(embed_dim * 2) for _ in range(depths[1])]

        self.stage3 = [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 4, kernel_size = 2, strides = 2)]
        self.stage3_block = [ConvNeXTBlock(embed_dim * 4) for _ in range(depths[2])]

        self.stage4 = [
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 8, kernel_size = 2, strides = 2)]
        self.stage4_block = [ConvNeXTBlock(embed_dim * 8) for _ in range(depths[3])]

        self.pool = tf.keras.layers.GlobalAvgPool2D()
        self.norm_layer2 = tf.keras.layers.LayerNormalization()
        self.class_output_layer = tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'class_predictions')


    def call(self, x):
        """Forward pass of the ConvNeXT network."""
        for layer in self.stage1:
            x = layer(x)
        for layer in self.stage1_block:
            x = layer(x)
        for layer in self.stage2:
            x = layer(x)
        for layer in self.stage2_block:
            x = layer(x)
        for layer in self.stage3:
            x = layer(x)
        for layer in self.stage3_block:
            x = layer(x)
        for layer in self.stage4:
            x = layer(x)
        for layer in self.stage4_block:
            x = layer(x)
        output = self.pool(x)
        if self.include_top:
            output = self.norm_layer2(output)
            output = self.class_output_layer(output)

        return output
