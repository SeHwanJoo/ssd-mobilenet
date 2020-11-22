import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2


class mobilenet_v2(tf.keras.Model):
    def __init__(self):
        super(mobilenet_v2, self).__init__()
        self.model = MobileNetV2(
            input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights='imagenet',
            input_tensor=None, pooling=None, classes=21,
        )
        self.model.summary()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(0, 155):
            if i == 119:
                branch_1 = self.model.get_layer(index=i)(x)

            if i in [18, 36, 63, 98, 125]:
                y = self.model.get_layer(index=i)(x)

            if i in [27, 54, 90, 116, 143]:
                x = self.model.get_layer(index=i)([y, x])
            elif i in [45, 72, 81, 107, 134]:
                y = self.model.get_layer(index=i)([y, x])
                x = y
            else:
                x = self.model.get_layer(index=i)(x)

        return branch_1, x


    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
        self.summary()

