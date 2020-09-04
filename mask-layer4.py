import tensorflow as tf

# 情况四
# 让mask传递到下一层，自定义层默认不会传递mask到下一层

class ReluActivation(tf.keras.layers.Layer):
    def __init__(self, supports_masking, **kwargs):
        super(ReluActivation, self).__init__(**kwargs)
        # 让mask传递到下一层，自定义层默认不会传递mask到下一层
        self.supports_masking = supports_masking

    def call(self, inputs):
        return tf.nn.relu(inputs)

def test(supports_masking):
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    x = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
    x = ReluActivation(supports_masking)(x)  # Will pass the mask along
    print("Mask found:", x._keras_mask)
    outputs = tf.keras.layers.LSTM(32)(x)  # Will receive the mask

    model = tf.keras.Model(inputs, outputs)

# 不会出错
test(True)

print("=" * 10)
try:
    test(False) 
except AttributeError as err:
    # 获得 'Tensor' object has no attribute '_keras_mask'
    print(err)
