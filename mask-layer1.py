import tensorflow as tf

# 情况一
# 需要消化mask
# 以AttentionPooling1D为例

class MaskingSoftmax(tf.keras.layers.Layer):

    def call(self, inputs, mask=None):
        if mask is None:
            broadcast_mask = 1
        else:
            broadcast_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        inputs_exp = tf.exp(inputs) * broadcast_mask
        inputs_sum = tf.reduce_sum(inputs * broadcast_mask, axis=1, keepdims=True)
        return inputs_exp / inputs_sum

class AttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, h_dim, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.kernel_initializer = kernel_initializer
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.h_dim, 
            use_bias=False, 
            kernel_initializer=self.kernel_initializer, 
            activation="tanh"
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1, 
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
        x = tf.math.softmax(x, 1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(x * x0, 1)
        return x

    def compute_output_shape(self, input_shape):
        return (None, self.h_dim)

inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = tf.keras.layers.Embedding(input_dim=100000, output_dim=128, mask_zero=True)(inputs)
x = AttentionPooling1D(128)(x)
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
maxlen = 256
x_train = tf.keras.preprocessing.sequence.pad_sequences(
    x_train, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0)

x_test = tf.keras.preprocessing.sequence.pad_sequences(
    x_test, 
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model.fit(x_train, y_train, epochs=10, batch_size=32)
model.evaluate(x_test, y_test)
