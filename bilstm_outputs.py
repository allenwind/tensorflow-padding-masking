import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 双向 LSTM 使用 mask 隐藏 padding 的内容
# 同时使用 post pre 方法 padding 以观察它们的不同效果

n_steps = 7
n_features = 2
pad_size = 3

a = np.random.normal(size=(n_steps, n_features))
b = np.zeros((pad_size, n_features))
x = np.vstack([b, a, b])
X = x[np.newaxis, :]

inputs = tf.keras.Input((n_steps + 2 * pad_size, n_features))
x = inputs
x = tf.keras.layers.Masking(mask_value=0.0)(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)

model = tf.keras.Model(inputs, x)
model.compile(optimizer="adam", loss="mse")
print(X)
y = model.predict(X)
print(y)

# 结论
# 当 padding 为 post mask 之后时间步的值都为 0
# 当 padding 为 pre mask 之前的时间步的值都为 0

"""
[[[ 0.          0.        ]
  [ 0.          0.        ]
  [ 0.          0.        ]
  [-0.02691833 -0.16752568]
  [-0.09884746 -0.03001245]
  [-0.21271683  0.04990812]
  [ 0.08978248 -0.0264868 ]
  [-0.021124   -0.11965897]
  [ 0.00110621  0.12473131]
  [-0.08888234  0.01038382]
  [ 0.          0.        ]
  [ 0.          0.        ]
  [ 0.          0.        ]]]
"""
