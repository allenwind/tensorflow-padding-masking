import tensorflow as tf

# 情况二
# 把mask传入内置层


class LSTMClassifier(tf.keras.layers.Layer):

    def __init__(self, h_dims=128, input_dim=100000, output_dim=128, **kwargs):
        super(LSTMClassifier, self).__init__(**kwargs)
        self.h_dims = h_dims
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            mask_zero=True # 指定mask值
        )
        self.lstm = tf.keras.layers.LSTM(self.h_dims)
    
    def call(self, inputs, mask=None):
        x = self.embedding(inputs)
        # 计算mask
        mask = self.embedding.compute_mask(inputs)
        return self.lstm(x, mask=mask) # lstm会忽略padding值
    
    def compute_output_shape(self, input_shape):
        return (None, self.h_dims)

inputs = tf.keras.Input(shape=(None,), dtype="int32")
x = LSTMClassifier(h_dims=128)(inputs)
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
