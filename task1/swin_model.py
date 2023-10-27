import tensorflow as tf
import os
import numpy as np
from swin_utils import window_partition, window_reverse, StochasticDepth

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

class PatchAndEmbed(tf.keras.layers.Layer):

    """ 
    Divide the image into fixed-size patches and linearly embed each patch. 

    Args:
        window_size (int): The size of each window in the output.
        patch_size (int): The size of each patch. Default is 4.
        embed_dim (int): The embedding dimension. Default is 96.
    """

    def __init__(self, window_size, patch_size = 4, embed_dim = 96):
        super(PatchAndEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Conv2D(embed_dim, patch_size, patch_size)
        self.norm = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.window_size = window_size

    def call(self, input):
        """ Forward pass for PatchAndEmbed."""
        x = self.proj(input)
        x = tf.reshape(x, [-1, self.window_size * self.window_size, self.embed_dim])
        x = self.norm(x)
        return x


class PatchMerging(tf.keras.layers.Layer):
    """ 
    Layer that merges neighboring patches.

    Args:
        input_res (tuple of ints): The resolution of input tensor.
        embed_dim (int): The embedding dimension.
    """

    def __init__(self, input_res, embed_dim):
        super(PatchMerging, self).__init__()
        self.input_res = input_res
        self.embed_dim = embed_dim
        self.reduction = tf.keras.layers.Dense(2 * embed_dim, use_bias = False)
        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon = 1e-5)

    def call(self, input):
        """ Forward pass for PatchMerging."""
        height, width = self.input_res
        _, _ , C = input.get_shape().as_list()
        x = tf.reshape(input, [-1, height, width, C])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis = -1)
        x = tf.reshape(x, [-1, (height // 2) * (width // 2), 4 * C])
        x = self.norm_layer(x)
        x = self.reduction(x)
        
        return x
    
class WindowAttention(tf.keras.layers.Layer):
    """ 
    Multi-head self attention mechanism for Swin Transformer.

    Args:
        embed_dim (int): The embedding dimension.
        window_size (tuple of ints): The size of window used.
        num_heads (int): Number of attention heads.
        attn_dropout_rate (float): Dropout rate for dropout layer after attention layers. Default is 0.
        proj_dropout_rate (float): Dropout rate for dropout layer after projection layers. Default is 0.
    """

    def __init__(self, 
                embed_dim, 
                window_size, 
                num_heads, 
                attn_dropout_rate = 0.,
                proj_dropout_rate = 0.):
        super(WindowAttention, self).__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = self.embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(3 * embed_dim, use_bias=True)
        self.projection = tf.keras.layers.Dense(embed_dim)  
        self.attn_Dropout = tf.keras.layers.Dropout(attn_dropout_rate) 
        self.proj_Dropout = tf.keras.layers.Dropout(proj_dropout_rate)

    def build(self, input_shape):
        """ Build relative positional encoding. """
        self.rel_pos_table = self.add_weight(shape = ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                           initializer = tf.keras.initializers.Zeros(),
                                                           trainable = True)
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing = 'ij'))
        coords = tf.reshape(coords, (2, -1))
        rel_coords = coords[:, :, None] - coords[:, None, :]
        rel_coords = tf.transpose(rel_coords, [1, 2, 0])
        rel_coords = rel_coords.numpy()
        rel_coords[:, :, 0] += self.window_size[0] - 1  
        rel_coords[:, :, 1] += self.window_size[1] - 1
        rel_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        rel_pos_index = rel_coords.sum(-1)
        self.rel_pos_index = tf.Variable(
            initial_value = tf.convert_to_tensor(rel_pos_index), trainable = False
        ) 


    def call(self, input, mask = None):
        """ Forward pass for WindowAttention."""
        _, num_patches, C = input.get_shape().as_list()
        qkv = self.qkv(input)
        qkv = tf.reshape(qkv, [3, -1, self.num_heads, num_patches, C // self.num_heads])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        attn = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]))
        rel_pos_bias = tf.gather(self.rel_pos_table, tf.reshape(self.rel_pos_index, [-1]))
        rel_pos_bias = tf.reshape(rel_pos_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        rel_pos_bias = tf.transpose(rel_pos_bias, [2, 0 ,1])
        rel_pos_bias = tf.expand_dims(rel_pos_bias, axis = 0)
        attn += rel_pos_bias
        if mask is not None:
            num_windows = mask.get_shape()[0]
            attn = tf.reshape(attn, [-1, num_windows, self.num_heads, num_patches, num_patches]) + tf.cast(tf.expand_dims(tf.expand_dims(mask, 1), 0), tf.float32)
            attn = tf.reshape(attn, [-1, self.num_heads, num_patches, num_patches])
            attn = tf.keras.activations.softmax(attn)
        else:
            attn = tf.keras.activations.softmax(attn)

        attn = self.attn_Dropout(attn)
        attn = tf.transpose((attn @ v), [0, 2, 1, 3])
        attn = tf.reshape(attn, [-1, num_patches, C])
        x = self.projection(attn)
        x = self.proj_Dropout(x)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    """
    Implements a transformer block with multi-head self attention mechanism.
    Uses a shifted window-based attention for capturing local and global information.
    
    Args:
        embed_dim(int): The embedding dimension.
        input_res(tuple of ints): The resolution of the input tensor.
        num_heads(int): Number of attention heads for the multi-head self attention mechanism.
        window_size(int): The size of the window to be used for the window-based attention. Default is 7.
        window_shift(int): Shift size for the attention window. Default is 0.
        attn_drop(float): Dropout rate for the dropout layer after the attention mechanism. Default is 0.
        proj_drop(float): Dropout rate for the dropout layer afterprojection. Default is 0.
    """

    def __init__(self, 
                embed_dim, 
                input_res,
                num_heads, 
                window_size = 7, 
                window_shift = 0, 
                attn_drop = 0.,
                proj_drop = 0.,
                ):
        
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.input_res = input_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_shift = window_shift


        self.norm_layer1 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.norm_layer2 = tf.keras.layers.LayerNormalization(epsilon = 1e-5)
        self.attn = WindowAttention(embed_dim, (self.window_size, self.window_size), num_heads, attn_drop, proj_drop)
        self.drop_path = StochasticDepth(0.2)
        self.mlp_fc1 = tf.keras.layers.Dense(units=(4 * embed_dim))
        self.mlp_drop1 = tf.keras.layers.Dropout(proj_drop)
        self.mlp_fc2 = tf.keras.layers.Dense(embed_dim)
        self.mlp_drop2 = tf.keras.layers.Dropout(proj_drop)
        if min(self.input_res) < self.window_size:
            self.window_shift = 0
            self.window_size = min(self.input_res)

    def build(self, input_shape):
        """
        Builds the layer by initializing a mask for the shifted window-based attention if the window shift is greater than 0.
        """
        if self.window_shift > 0:
            height, width = self.input_res
            mask = np.zeros((1, height, width, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.window_shift),
                slice(-self.window_shift, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.window_shift),
                slice(-self.window_shift, None),
            )
            count = 0
            for h in h_slices: 
                for w in w_slices:
                    mask[:, h, w, :] = count
                    count += 1
            mask = tf.convert_to_tensor(mask)
            mask_windows = window_partition(mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            mask = tf.where(mask != 0, -100.0, mask)
            mask = tf.where(mask == 0, 0.0, mask)
            self.mask = tf.Variable(initial_value=mask, trainable=False)
        else:
            self.mask = None

    def call(self, input, training = None):
        """ Forward pass for TransformerBlock."""
        height, width = self.input_res
        _, num_patches, C = input.get_shape().as_list()
        skip_connection = input
        x = self.norm_layer1(input)
        x = tf.reshape(x, [-1, height, width, C])
        if self.window_shift > 0:
            shifted_x = tf.roll(
                x, shift = [-self.window_shift, -self.window_shift], axis = [1, 2]
            )
        else:
            shifted_x = x

        partitioned_windows = window_partition(shifted_x, self.window_size)
        partitioned_windows = tf.reshape(partitioned_windows, [-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(partitioned_windows, self.mask)
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        x = window_reverse(attn_windows,  self.window_size, height, width, C)
        if self.window_shift > 0:
            x = tf.roll(
                x, shift = [self.window_shift, self.window_shift], axis = [1, 2]
            )
        x = tf.reshape(x, [-1, height * width, C])
        x = self.drop_path(x, training = training) + skip_connection
        skip_connection = x
        x = self.norm_layer2(x)
        x = self.mlp_fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.mlp_drop1(x)
        x = self.mlp_fc2(x)
        x = self.mlp_drop2(x)
        x = self.drop_path(x, training = training) + skip_connection
        return x
    
class StageTransformerBlocks(tf.keras.layers.Layer):
    """
    Implements a series of Transformer blocks for a specific stage in the architecture.
    
    Args:
        depth(int): Number of Transformer blocks in this stage.
        input_dim(int): Dimension of the input tensor.
        embed_dim(int): The embedding dimension.
        num_heads(int): Number of attention heads for the multi-head self attention mechanism.
        attn_drop(float): Dropout rate for the dropout layer after the attention mechanism. Default is 0.
        proj_drop(float): Dropout rate for the dropout layer after projection. Default is 0.
        window_size(int): The size of the window to be used for the window-based attention. Default is 7.
    """

    def __init__(self, depth, input_dim, embed_dim, num_heads, attn_drop = 0., proj_drop = 0., window_size = 7):
        super(StageTransformerBlocks, self).__init__()
        self.transformers = [
            TransformerBlock(
                embed_dim = embed_dim,
                input_res = (input_dim, input_dim),
                num_heads = num_heads,
                window_size = window_size,
                window_shift = 0 if i % 2 == 0 else window_size // 2,
                attn_drop = attn_drop,
                proj_drop = proj_drop
            )
            for i in range(depth)
        ]

    def call(self, x):
        """ Forward pass for StageTransformerBlock."""
        for layer in self.transformers:
            x = layer(x)
        return x


class SwinTransformer(tf.keras.Model):
    """
    Swin Transformer architecture. This model divides an image into patches, embeds them, and 
    uses a series of transformer blocks for feature extraction, followed by a classification head.
    
    Args:
        num_classes(int): Number of output classes for classification.
        input_dim(int): Dimension of the input image. Default is 224.
        patch_size(int): Number of pixels along one dimension of each patch. Default is 4.
        embed_dim(int): Embedding dimension for the patches. Default is 96.
        depth(list of ints): List of number of Transformer blocks for each stage.
        num_heads(list of ints): List of number of attention heads for each stage.
        window_size(int): The size of the window to be used for the window-based attention in each block. Default is 7.
        attn_drop(float): Dropout rate for the dropout layer after the attention mechanism. Default is 0.
        proj_drop(float): Dropout rate for the dropout layer after projection. Default is 0.
    """


    def __init__(self,
                num_classes,
                input_dim = 224, 
                patch_size = 4, 
                embed_dim = 96,
                depth = [2, 2, 6, 2],
                num_heads = [3, 6, 12, 24],
                window_size = 7,
                attn_drop_rate = 0.,
                proj_drop_rate = 0.,
                include_top = True
                ):
        super(SwinTransformer, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.include_top = include_top
        self.patch_embed = PatchAndEmbed(input_dim // patch_size, patch_size, embed_dim)
       
        self.transformers_stage1 = StageTransformerBlocks(depth[0], 
                                                        input_dim // patch_size, 
                                                        embed_dim, 
                                                        num_heads[0],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.merge1 = PatchMerging((input_dim // patch_size, input_dim // patch_size), embed_dim)
        self.transformers_stage2 = StageTransformerBlocks(depth[1], 
                                                        input_dim // (2 * patch_size), 
                                                        embed_dim * 2, 
                                                        num_heads[1],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.merge2 = PatchMerging((input_dim // (2 * patch_size), input_dim // (2 * patch_size)), 2 * embed_dim)
        self.transformers_stage3 = StageTransformerBlocks(depth[2], 
                                                        input_dim // (4 * patch_size), 
                                                        embed_dim * 4, 
                                                        num_heads[2],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.merge3 = PatchMerging((input_dim // (4 * patch_size), input_dim // (4 * patch_size)), 4 * embed_dim)
        self.transformers_stage4 = StageTransformerBlocks(depth[3], 
                                                        input_dim // (8 * patch_size), 
                                                        embed_dim * 8,
                                                        num_heads[3],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.class_output_layer = tf.keras.layers.Dense(num_classes, activation = 'softmax', name = 'class_predictions')
    
    def call(self, input):
        """ Forward pass for SwinTransformer."""
        x = self.patch_embed(input)
        x = self.transformers_stage1(x)
        x = self.merge1(x)
        x = self.transformers_stage2(x)
        x = self.merge2(x)
        x = self.transformers_stage3(x)
        x = self.merge3(x)
        x = self.transformers_stage4(x)
        output = self.pool(x)
        if self.include_top:
            output = self.class_output_layer(x)

        return output

