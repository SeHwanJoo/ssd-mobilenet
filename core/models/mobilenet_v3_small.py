import tensorflow as tf
from core.models.mobilenet_v3_block import BottleNeck, h_swish
from configuration import NUM_CLASSES


class MobileNetV3Small(tf.keras.Model):
    def __init__(self):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=32, exp_size=32, out_size=64, s=2, is_se_existing=True, NL="RE", k=3)
        self.bneck2 = BottleNeck(in_size=64, exp_size=64, out_size=128, s=2, is_se_existing=False, NL="RE", k=3)
        self.bneck3 = BottleNeck(in_size=128, exp_size=128, out_size=128, s=1, is_se_existing=False, NL="RE", k=3)
        self.bneck4 = BottleNeck(in_size=128, exp_size=128, out_size=256, s=2, is_se_existing=True, NL="RE", k=5)
        self.bneck5 = BottleNeck(in_size=256, exp_size=256, out_size=256, s=1, is_se_existing=True, NL="RE", k=5)
        self.bneck6 = BottleNeck(in_size=256, exp_size=256, out_size=256, s=1, is_se_existing=True, NL="RE", k=5)
        self.bneck7 = BottleNeck(in_size=256, exp_size=256, out_size=512, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck8 = BottleNeck(in_size=512, exp_size=512, out_size=512, s=1, is_se_existing=True, NL="HS", k=5)
        self.bneck9 = BottleNeck(in_size=512, exp_size=512, out_size=512, s=1, is_se_existing=True, NL="HS", k=5)
        # self.bneck10 = BottleNeck(in_size=512, exp_size=512, out_size=512, s=1, is_se_existing=True, NL="RE", k=3)
        # self.bneck11 = BottleNeck(in_size=512, exp_size=512, out_size=512, s=1, is_se_existing=True, NL="RE", k=3)
        # self.bneck11 = BottleNeck(in_size=512, exp_size=512, out_size=512, s=1, is_se_existing=True, NL="RE", k=3)
        self.bneck10 = BottleNeck(in_size=512, exp_size=512, out_size=1024, s=2, is_se_existing=True, NL="HS", k=5)
        self.bneck11 = BottleNeck(in_size=1024, exp_size=1024, out_size=512, s=1, is_se_existing=True, NL="HS", k=5)

        self.conv2 = tf.keras.layers.Conv2D(filters=576,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7),
                                                        strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=1280,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=NUM_CLASSES,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same",
                                            activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = h_swish(x)

        x = self.bneck1(x, training=training)
        x = self.bneck2(x, training=training)
        x = self.bneck3(x, training=training)
        x = self.bneck4(x, training=training)
        x = self.bneck5(x, training=training)
        x = self.bneck6(x, training=training)
        x = self.bneck7(x, training=training)
        x = self.bneck8(x, training=training)
        branch_1 = self.bneck9(x, training=training)
        x = self.bneck10(branch_1, training=training)
        x = self.bneck11(x, training=training)
        # x = self.bneck12(branch_1, training=training)
        # x = self.bneck13(x, training=training)

        # x = self.conv2(x)
        # x = self.bn2(x, training=training)
        # x = h_swish(x)
        # x = self.avgpool(x)
        # x = self.conv3(x)
        # x = h_swish(x)
        # x = self.conv4(x)

        return branch_1, x