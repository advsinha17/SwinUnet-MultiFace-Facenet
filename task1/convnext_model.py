import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class ConvNeXTBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim):

        super(ConvNeXTBlock, self).__init__()
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size = 7, padding = "same")
        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.pointwise_conv1 = tf.keras.layers.Conv2D(filters = 4 * embed_dim, kernel_size = 1)
        self.gelu = tf.keras.layers.Activation(tf.keras.activations.gelu)
        self.pointwise_conv2 = tf.keras.layers.Conv2D(filters = embed_dim, kernel_size = 1)

    def call(self, input):
        x = self.depthwise_conv(input)
        x = self.norm_layer(input)
        x = self.pointwise_conv1(x)
        x = self.gelu(x)
        x = self.pointwise_conv2(x)
        x += input

        return x

class ConvNeXT(tf.keras.Model):

    def __init__(self, num_labels, embed_dim = 96, depths = [3, 3, 9, 3]):
        
        super(ConvNeXT, self).__init__()
        self.patchify_stem = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = embed_dim, kernel_size = 4, strides = 4),
            tf.keras.layers.LayerNormalization()
        ])
        self.stage1 = tf.keras.Sequential([
            ConvNeXTBlock(embed_dim)
            for _ in range(depths[0])
        ])
        self.stage2 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 2, kernel_size = 2, strides = 2)] + 
            [ ConvNeXTBlock(embed_dim * 2) for _ in range(depths[1])
            ])
        self.stage3 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 4, kernel_size = 2, strides = 2)] + 
            [ ConvNeXTBlock(embed_dim * 4) for _ in range(depths[2])
            ])
        self.stage4 = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(filters = embed_dim * 8, kernel_size = 2, strides = 2)] + 
            [ ConvNeXTBlock(embed_dim * 8) for _ in range(depths[3])
            ])
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.norm_layer = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(num_labels)


    def call(self, x):
        for layer in self.patchify_stem.layers:
            x = layer(x)
        for layer in self.stage1.layers:
            x = layer(x)
        for layer in self.stage2.layers:
            x = layer(x)
        for layer in self.stage3.layers:
            x = layer(x)
        for layer in self.stage4.layers:
            x = layer(x)
        x = self.pool(x)
        x = self.norm_layer(x)
        output = self.output_layer(x)

        return output