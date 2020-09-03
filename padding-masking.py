import itertools
import numpy as np
import tensorflow as tf

def batch_padding(X, padding=0):
    maxlen = max([len(x) for x in X])
    padded_X = np.array([
        np.concatenate([x, [padding] * (maxlen - len(x))])
        if len(x) < maxlen else x for x in X
    ])
    return padded_X

class Word2IdTransformer:
    """字转id"""

    def __init__(self):
        self.word2id = {}
        self.UNKNOW = 1 # pad is 0

    def fit(self, X):
        vocab = set(itertools.chain(*X))
        for i, w in enumerate(vocab, start=2):
            self.word2id[w] = i

    def transform(self, X):
        r = []
        for sample in X:
            s = []
            for w in sample:
                s.append(self.word2id.get(w, self.UNKNOW))
            r.append(s)
        return r

    def __len__(self):
        return len(self.word2id) + 1

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

padded_X = tf.keras.preprocessing.sequence.pad_sequences(
    X, padding="post"
)
print(padded_X)

# or
print(batch_padding(X))


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
    tf.tile(tf.expand_dims(padded_X, axis=-1), [1, 1, 8]), tf.float32
)
masked_embedding = masking(unmasked_embedding)
print(masked_embedding._keras_mask)
