import tensorflow as tf
from tensorflow import keras


class ConvBNRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, strides=1, padding='SAME', weight_decay=0.0005, rate=0.4, drop=True):
        super(ConvBNRelu, self).__init__()
        self.drop = drop
        self.conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                        padding=padding, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.dropOut = keras.layers.Dropout(rate=rate)

    def call(self, inputs, training=False):
        layer = self.conv(inputs)
        layer = tf.nn.relu(layer)
        layer = self.batchnorm(layer)
        if self.drop:
            layer = self.dropOut(layer)

        return layer


class VGG16(tf.keras.Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = ConvBNRelu(filters=64, kernel_size=[3, 3], rate=0.3)
        self.conv2 = ConvBNRelu(filters=64, kernel_size=[3, 3], drop=False)
        self.maxPooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3])
        self.conv4 = ConvBNRelu(filters=128, kernel_size=[3, 3], drop=False)
        self.maxPooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv5 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv6 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        # self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3])
        self.conv7 = ConvBNRelu(filters=256, kernel_size=[3, 3], drop=False)
        self.maxPooling3 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv11 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv12 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        # self.conv13 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv14 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)
        self.maxPooling5 = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv15 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv16 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        # self.conv17 = ConvBNRelu(filters=512, kernel_size=[3, 3])
        self.conv18 = ConvBNRelu(filters=512, kernel_size=[3, 3], drop=False)

    def call(self, inputs, training=None, **kwargs):
        net = self.conv1(inputs, training=training)
        net = self.conv2(net, training=training)
        net = self.maxPooling1(net)
        net = self.conv3(net, training=training)
        net = self.conv4(net, training=training)
        net = self.maxPooling2(net)
        net = self.conv5(net, training=training)
        net = self.conv6(net, training=training)
        # net = self.conv7(net, training=training)
        net = self.maxPooling3(net)
        net = self.conv11(net, training=training)
        net = self.conv12(net, training=training)
        # branch_1 = self.conv13(net, training=training)
        branch_1 = self.conv14(net, training=training)
        branch_2 = self.maxPooling5(branch_1)
        branch_2 = self.conv15(branch_2, training=training)
        branch_2 = self.conv16(branch_2, training=training)
        # branch_2 = self.conv17(branch_2, training=training)
        branch_2 = self.conv18(branch_2, training=training)
        return branch_1, branch_2
