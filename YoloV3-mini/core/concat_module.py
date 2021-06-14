import common
import tensorflow as tf


class concat_layer_1(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(concat_layer_1, self).__init__()
        self.conv = common.Conv_set(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.upsampling(x)
        return inputs, x


class concat_layer_2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(concat_layer_2, self).__init__()
        self.conv = common.Conv_set(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, input_1, input_2, training=None, **kwargs):
        x = self.conv(input_1)
        output_1 = self.concat([x, input_2])
        output_2 = self.upsampling(output_1)
        return output_1, output_2


class concat_layer_3(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(concat_layer_3, self).__init__()
        self.conv = common.Conv_set(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides)
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, input_1, input_2, training=None, **kwargs):
        x = self.conv(input_1)
        x = self.concat([x, input_2])
        return x


class concat_module(tf.keras.Model):
    def __init__(self):
        super(concat_module, self).__init__()
        self.layer_1 = concat_layer_1(filters=512, kernel_size=(1, 1), strides=1)
        self.layer_2 = concat_layer_2(filters=256, kernel_size=(3, 3), strides=1)
        self.layer_3 = concat_layer_3(filters=128, kernel_size=(3, 3), strides=1)
        self.conv = common.Conv_set(filters=256, kernel_size=(1, 1), strides=1)

    def call(self, input_1, input_2, input_3, training=None, **kwargs):
        output_1,x = self.layer_1(input_1)
        output_2, x = self.layer_2(input_2, x)
        x = self.layer_3(input_3, x)
        output_3 = self.conv(x)
        # print(output_1.shape, output_2.shape, output_3.shape)
        return output_1, output_2, output_3
