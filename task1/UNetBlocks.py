import tensorflow as tf



class DoubleConv(tf.keras.layers.Layer):

    """

    Class that returns the double convultional layers required for each block of a UNet.

    Args:
        n_filters: Number of output features in the convolution

    """
    def __init__(self, n_filters):
        super(DoubleConv, self).__init__()

        # conv layer
        self.conv = tf.keras.layers.Conv2D(
                    n_filters, 
                    kernel_size=(3, 3), 
                    padding='same',
                    kernel_initializer='he_normal'
                )
        
        # batch norm layer
        self.bn   = tf.keras.layers.BatchNormalization()
    
    def call(self, input_tensor, training=False):

        """

        Returns the output of two Convolution -> Batch Normalization -> ReLU units given the input tensor.

        Args:
            input_tensor: Input tensor for the DoubleConv unit.
            training: True if model is being trained, False otherwise.

        Returns:
            x: The output of the DoubleConv unit.

        """
        
        # First DoubleConv unit, comprises of Convolution layer -> Batch Normalization layer -> ReLU activation
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        # Second DoubleConv unit
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x


class UpsamplingBlock(tf.keras.layers.Layer):

    """

    Class that defines one Upsampling/Decoding unit of the UNet.

    Args:
        n_filters: Number of output filters for the DoubleConv unit of each downsampling unit.
    
    """

    def __init__(self, n_filters = 32, last = False):

        super(UpsamplingBlock, self).__init__()

        # DoubleConv layer
        self.double_conv = DoubleConv(n_filters)

        # Transpose convolution layer
        self.transpose = tf.keras.layers.Conv2DTranspose(n_filters,
                                                         kernel_size = (3, 3),
                                                         strides = 2 if not last else 4,
                                                         padding = 'same')
        
        # Concatenation layer
        self.concatenate = tf.keras.layers.Concatenate(axis = 3)

    def call(self, input_tensor, skip_input, training = False):

        """

        Returns output for the next upsampling unit.

        Args:
            input_tensor: Inpur tensor to the unpsampling unit.
            skip_input: Input from corresponding downsampling block passed through a skip connection.
            training: True if model is being trained, False otherwise.
        
        Returns:
            x: Output of each unsampling unit.

        """
        
        # Transpose Convolution.
        up = self.transpose(input_tensor)

        # Concatenation of transpose convolution output with skip_input before passing to DoubleConv layer.
        x = self.concatenate([up, skip_input])

        # DoubleConv layer.
        x = self.double_conv(x, training = training)

        return x