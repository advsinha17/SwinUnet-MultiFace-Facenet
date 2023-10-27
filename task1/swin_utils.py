import tensorflow as tf


def window_partition(x, window_size):
    """
    Partitions the input tensor into non-overlapping windows.

    Args:
        x (tf.Tensor): Input tensor of shape [B, H, W, C], where B is batch size, H is height, 
                     W is width, and C is number of channels.
        window_size (int): Size of each square window to partition.

    Returns:
        windows (tf.Tensor): Tensor of windows of shape [num_windows, window_size, window_size, C].
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Reconstructs the tensor from its partitioned windows.

    Args:
        windows (tf.Tensor): Tensor of windows of shape [num_windows, window_size, window_size, C].
        window_size (int): Size of each square window.
        H, W, C (int): Height, Width, and Channel dimension for the reconstructed tensor.

    Returns:
        x (tf.Tensor): Reconstructed tensor of shape [B, H, W, C].
    """
    x = tf.reshape(windows, [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, C])
    return x

class StochasticDepth(tf.keras.layers.Layer):
    """
    Implements the Stochastic Depth layer, a form of structured dropout.
    Reference: https://arxiv.org/abs/1603.09382

    Args:
        dropout_prob (float): Probability of dropping out a given layer.
        scale_by_keep (bool): If True, scales the activations such that the expected sum remains constant.
    """

    def __init__(self, dropout_prob, scale_by_keep = True):
        super(StochasticDepth, self).__init__()
        self.scale_by_keep = scale_by_keep
        self.drop_prob = dropout_prob

    def call(self, input, training=None):
        if not training:
            return input
        
        shape = (tf.shape(input)[0],) + (1,) * (len(tf.shape(input)) - 1)
        random_tensor = tf.random.uniform(shape, minval = 0, maxval = 1)
        random_tensor = tf.cast(random_tensor < (1 - self.drop_prob), input.dtype)
        if 1 - self.drop_prob > 0 and self.scale_by_keep:
            random_tensor /= (1 - self.drop_prob)
        return input * random_tensor
