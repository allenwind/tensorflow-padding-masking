import tensorflow as tf
import numpy as np

class SimpleEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        embeddings_initializer="uniform",
        mask_zero=False, **kwargs):
        super(SimpleEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            dtype="float32",
        )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.sequence_mask(inputs)


embedding = SimpleEmbedding(10, 32, mask_zero=True)
x = np.random.random((3, 10)) * 9
x = x.astype("int32")

y = embedding(x)
mask = embedding.compute_mask(x)
print(y._keras_mask)
print(mask)
