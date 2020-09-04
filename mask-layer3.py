import tensorflow as tf
import numpy as np

# 情况三
# (samples, timesteps, features)中timesteps维度被改变，mask需要相应改变

class TimeseriesSplit(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.split(inputs, 2, axis=1)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        # mask需要做相应的改变
        return tf.split(mask, 2, axis=1)

X = [
    [711, 632, 71],
    [73, 8, 3215, 55, 927],
    [83, 91, 1, 645, 1253, 927],
]

padded_X = tf.keras.preprocessing.sequence.pad_sequences(
    X, padding="post"
)
print(padded_X)
masking_layer = tf.keras.layers.Masking()
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_X, axis=-1), [1, 1, 10]), tf.float32
)

masked_embedding = masking_layer(unmasked_embedding)
first_half, second_half = TimeseriesSplit()(masked_embedding)
print(first_half._keras_mask)
print(second_half._keras_mask)

