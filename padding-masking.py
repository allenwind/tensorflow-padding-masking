import itertools
import numpy as np
import tensorflow as tf
from preprocessing import Word2IdTransformer

def batch_padding(X, padding=0):
    maxlen = max([len(x) for x in X])
    padded_X = np.array([
        np.concatenate([x, [padding] * (maxlen - len(x))])
        if len(x) < maxlen else x for x in X
    ])
    return padded_X

##### padding #####
seqs = [
  ["Hello", "world", "!"],
  ["How", "are", "you", "doing", "today"],
  ["The", "weather", "will", "be", "nice", "tomorrow"],
]

wtrans = Word2IdTransformer()
wtrans.fit(seqs)
X = wtrans.transform(seqs) # 转换为ID
print(X)

# 示例样本
seqs = [
  np.arange(1, 7),
  np.arange(4, 11),
  np.arange(1, 8),
]

# 1
padded_X = tf.keras.preprocessing.sequence.pad_sequences(
    X, padding="post"
)
print(padded_X)

# 2
print(batch_padding(X))

# 3
# tf.data的batch_pad方法

def gen():
    for i in X:
        yield i

dl = tf.data.Dataset.from_generator(
    generator=gen,
    output_types=tf.int32
).padded_batch(
    batch_size=1,
    padded_shapes=[10]
)

for i in iter(dl):
    print(i.shape)

##### generate mask #####


# 1
mask = tf.not_equal(padded_X, 0)
print(mask)


# 2
lengths = [len(x) for x in X]
mask = tf.sequence_mask(lengths)
print(mask)


# 3
embedding = tf.keras.layers.Embedding(
    input_dim=20, output_dim=8, mask_zero=True)
masked_output = embedding(padded_X)
print(masked_output._keras_mask)


# 4
masking = tf.keras.layers.Masking(mask_value=0.0)
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_X, axis=-1), [1, 1, 8]), tf.float32 # (batch_size, timesteps, features)
)
masked_embedding = masking(unmasked_embedding)
print(masked_embedding._keras_mask)

# 5
mask = tf.greater(padded_X, 0)
print(mask)

# 6
mask = tf.math.logical_not(tf.math.equal(padded_X, 0))
print(mask)

# 7
mask_layer = tf.keras.layers.Masking(mask_value=0)
X = tf.expand_dims(padded_X, axis=-1)
mask_tensor = mask_layer.compute_mask(X)
print(mask_tensor)
