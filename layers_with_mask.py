import tensorflow as tf
from tensorflow.keras.layers import *

class MaskGlobalAveragePooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalAveragePooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x = x * mask
        x = tf.reduce_sum(x, axis=1)
        return x / (tf.reduce_sum(mask, axis=1) + 1e-12)

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        return tf.reduce_max(x, axis=1)

class MaskGlobalMaxMinPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxMinPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x1 = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x1 = tf.reduce_max(x1, axis=1)
        x2 = x + (1 - mask) * 1e12
        x2 = tf.reduce_min(x2, axis=1)
        return tf.concat([x1, x2], axis=1)

class AttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, h_dim, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim # 类似于CNN中的filters
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.h_dim,
            kernel_initializer=self.kernel_initializer,
            #kernel_regularizer="l2",
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            kernel_regularizer="l1", # 添加稀疏性
            kernel_constraint=tf.keras.constraints.non_neg(), # 添加非负约束
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        x = self.k_dense(inputs)
        x = self.o_dense(x)
        # 处理 mask
        x = x - (1 - mask) * 1e12
        # 权重归一化
        x = tf.math.softmax(x, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(x * x0, axis=1)
        return x

class MaskBiLSTM(tf.keras.layers.Layer):
    def __init__(self, hdims, **kwargs):
        super(MaskBiLSTM, self).__init__(**kwargs)
        self.hdims = hdims
        self.forward_lstm = LSTM(hdims, return_sequences=True)
        self.backend_lstm = LSTM(hdims, return_sequences=True)

    def reverse_sequence(self, x, mask):
        seq_len = tf.reduce_sum(mask, axis=1)[:, 0]
        seq_len = tf.cast(seq_len, tf.int32)
        x = tf.reverse_sequence(x, seq_len, seq_axis=1)
        return x

    def call(self, inputs, mask):
        x = inputs
        x_forward = self.forward_lstm(x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.backend_lstm(x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = tf.concat([x_forward, x_backward], axis=-1)
        x = x * mask
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.hdims * 2,)
