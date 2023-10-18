import tensorflow as tf
import os
import numpy as np
from swin_utils import window_partition, window_reverse

os.environ['TF_CPP_MIN_LG_LEVEL'] = "2"

class PatchPartition(tf.keras.layers.Layer):

    def __init__(self, patch_size = 4):
        super(PatchPartition, self).__init__();
        self.patch_size = patch_size

    def call(self, input):
        B, _, _, C = input.shape
        patches = tf.image.extract_patches(
            images = input,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID"
        )
        patch_dim = self.patch_size * self.patch_size * C
        patches = tf.reshape(patches, (B, -1, patch_dim))
        return patches
 

class PatchEmbedding(tf.keras.layers.Layer):

    def __init__(self, embed_dim = 96, norm_layer = None):
        super(PatchEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.norm = norm_layer


    def call(self, input):
        x = self.proj(input)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(tf.keras.layers.Layer):

    def __init__(self, input_res, embed_dim):
        super(PatchMerging, self).__init__()
        self.input_res = input_res
        self.embed_dim = embed_dim
        self.reduction = tf.keras.layers.Dense(2 * embed_dim, use_bias = False)
        self.norm_layer = tf.keras.layers.LayerNormalization()

    def call(self, input):
        height, weight = self.input_res
        B, _ , C = input.shape
        x = tf.reshape(input, [B, height, weight, C])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis = -1)
        x = tf.reshape(x, [B, -1, 4 * C])
        x = self.norm_layer(x)
        x = self.reduction(x)
        
        return x
    
class WindowAttention(tf.keras.layers.Layer):

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

        self.rel_pos_table = self.add_weight(shape = ((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
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
        self.qkv = tf.keras.layers.Dense(3 * embed_dim, use_bias=True)
        self.projection = tf.keras.layers.Dense(embed_dim)  
        self.attn_Dropout = tf.keras.layers.Dropout(attn_dropout_rate) 
        self.proj_Dropout = tf.keras.layers.Dropout(proj_dropout_rate)


    def call(self, input, mask = None):
        num_batch_windows, num_patches, C = input.shape
        qkv = self.qkv(input)
        qkv = tf.reshape(qkv, [num_batch_windows, num_patches, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        attn = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]))
        rel_pos_bias = tf.gather(self.rel_pos_table, tf.reshape(self.rel_pos_index, [-1]))
        rel_pos_bias = tf.reshape(rel_pos_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        rel_pos_bias = tf.transpose(rel_pos_bias, [2, 0 ,1])
        rel_pos_bias = tf.expand_dims(rel_pos_bias, axis = 0)
        attn += rel_pos_bias
        if mask is not None:
            num_windows = mask.shape[0]
            attn = tf.reshape(attn, [num_batch_windows // num_windows, num_windows, self.num_heads, num_patches, num_patches]) + tf.cast(tf.expand_dims(tf.expand_dims(mask, 1), 0), tf.float32)
            attn = tf.reshape(attn, [-1, self.num_heads, num_patches, num_patches])
            attn = tf.keras.activations.softmax(attn)
        else:
            attn = tf.keras.activations.softmax(attn)

        attn = self.attn_Dropout(attn)
        attn = tf.transpose((attn @ v), [0, 2, 1, 3])
        attn = tf.reshape(attn, [num_batch_windows, num_patches, C])
        x = self.projection(attn)
        x = self.proj_Dropout(x)
        return x


class TransformerBlock(tf.keras.layers.Layer):

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


        self.norm_layer1 = tf.keras.layers.LayerNormalization()
        self.norm_layer2 = tf.keras.layers.LayerNormalization()
        self.attn = WindowAttention(embed_dim, (self.window_size, self.window_size), num_heads, attn_drop, proj_drop)
        self.Mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * embed_dim),
            tf.keras.layers.Activation(tf.keras.activations.gelu),
            tf.keras.layers.Dropout(proj_drop),
            tf.keras.layers.Dense(embed_dim),
            tf.keras.layers.Dropout(proj_drop)
        ])
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

    def call(self, input):
        height, width = self.input_res
        B, num_patches, C = input.shape
        skip_connection = input
        x = self.norm_layer1(input)
        x = tf.reshape(x, [B, height, width, C])
        if self.window_shift > 0:
            shifted_x = tf.roll(
                x, shift = [-self.window_shift, self.window_shift], axis = [1, 2]
            )
        else:
            shifted_x = x

        partitioned_windows = window_partition(shifted_x, self.window_size)
        partitioned_windows = tf.reshape(partitioned_windows, [-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(partitioned_windows, self.mask)
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        x = window_reverse(attn_windows,  self.window_size, height, width)
        if self.window_shift > 0:
            x = tf.roll(
                x, shift = [self.window_shift, self.window_shift], axis = [1, 2]
            )
        x = tf.reshape(x, [B, num_patches, C])
        x += skip_connection
        skip_connection = x
        x = self.norm_layer2(x)
        x = self.Mlp(x)
        x += skip_connection
        return x
    
class StageTransformerBlocks(tf.keras.layers.Layer):

    def __init__(self, depth, input_dim, embed_dim, num_heads, attn_drop = 0., proj_drop = 0., window_size = 7):
        super(StageTransformerBlocks, self).__init__()
        self.transformers = tf.keras.Sequential([
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
        ])

    def call(self, x):
        for layer in self.transformers.layers:
            x = layer(x)
        return x


class SwinTransformer(tf.keras.Model):

    def __init__(self,
                num_labels,
                input_dim = 224, 
                patch_size = 4, 
                embed_dim = 96,
                depth = [2, 2, 6, 2],
                num_heads = [3, 6, 12, 24],
                norm_layer = None,
                window_size = 7,
                attn_drop_rate = 0.,
                proj_drop_rate = 0.
                ):
        super(SwinTransformer, self).__init__()

        self.partition = PatchPartition(patch_size)
        self.linear_embedding = PatchEmbedding(embed_dim, norm_layer)
        self.merge1 = PatchMerging((input_dim // 4, input_dim // 4), embed_dim)
        self.merge2 = PatchMerging((input_dim // 8, input_dim // 8), 2 * embed_dim)
        self.merge3 = PatchMerging((input_dim // 16, input_dim // 16), 4 * embed_dim)
        self.transformers_stage1 = StageTransformerBlocks(depth[0], 
                                                        input_dim // 4, 
                                                        embed_dim, 
                                                        num_heads[0],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.transformers_stage2 = StageTransformerBlocks(depth[1], 
                                                        input_dim // 8, 
                                                        embed_dim * 2, 
                                                        num_heads[1],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.transformers_stage3 = StageTransformerBlocks(depth[2], 
                                                        input_dim // 16, 
                                                        embed_dim * 4, 
                                                        num_heads[2],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.transformers_stage4 = StageTransformerBlocks(depth[3], 
                                                        input_dim // 32, 
                                                        embed_dim * 8,
                                                        num_heads[3],
                                                        attn_drop_rate,
                                                        proj_drop_rate,
                                                        window_size)
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(num_labels, activation = 'sigmoid')

    def call(self, input):
        x = self.partition(input)
        x = self.linear_embedding(x)
        x = self.transformers_stage1(x)
        x = self.merge1(x)
        x = self.transformers_stage2(x)
        x = self.merge2(x)
        x = self.transformers_stage3(x)
        x = self.merge3(x)
        x = self.transformers_stage4(x)
        x = self.pool(x)
        outputs = self.output_layer(x)

        return outputs