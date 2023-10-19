import tensorflow as tf


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, C])
    return x

class StochasticDepth(tf.keras.layers.Layer):

    def __init__(self, dropout_prob, scale_by_keep = True):
        super(StochasticDepth, self).__init__()
        self.scale_by_keep = scale_by_keep
        self.drop_prob = dropout_prob


    def call(self, input):
        shape = (tf.shape(input)[0],) + (1,) * (len(tf.shape(input)) - 1)
        random_tensor = tf.random.uniform(shape, minval = 0, maxval = 1)
        random_tensor = tf.cast(random_tensor < (1 - self.drop_prob), input.dtype)
        if 1 - self.drop_prob > 0 and self.scale_by_keep:
            random_tensor /= (1 - self.drop_prob)
        return input * random_tensor