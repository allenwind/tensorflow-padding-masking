import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 使用 mask 隐藏 padding 的内容

n_steps = 7
n_features = 2
pad_size = 3

a = np.random.normal(size=(n_steps, n_features))
mask = np.zeros((pad_size, n_features))
x = np.vstack([mask, a, mask]) # 模拟同时 post、pre padding
X = x[np.newaxis, :]

def gen_vary_sequences(size):
    seqs = []
    labels = []
    for _ in range(size):
        l = np.random.randint(10, 20)
        s = np.random.normal(size=(l, n_features))
        seqs.append(s)
        labels.append(np.mean(s))
    return seqs, labels

# X = pad_sequences(seqs, maxlen=n_steps, dtype="float32", padding="post", value=0)
# y = np.array(labels).reshape((len(seqs), 1))

input = tf.keras.Input((n_steps+2*pad_size, n_features))
x = input
x = tf.keras.layers.Masking(mask_value=0.0)(x)
x = tf.keras.layers.LSTM(2, return_sequences=True)(x)
x = tf.keras.layers.LSTM(2, return_sequences=True)(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)

model = tf.keras.Model(input, x)
model.compile(optimizer="adam", loss="mse")
print(X)
y = model.predict(X)
print(y)
print(y.shape)

# 结论
# 当 padding 为 post mask 之后时间步的值都不变了
# 当 padding 为 pre mask 之前的时间步的值都为0
# stacked 情况也一样

"""
  [ 0.18722317]
  [ 0.02635195]
  [ 0.02364353]
  [-0.13976887]
  [-0.08580425]
  [ 0.16069974]
  [ 0.10021626]
  [ 0.10021626] 0
  [ 0.10021626] 0
  [ 0.10021626] 0
"""
