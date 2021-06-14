import tensorflow as tf
import common


class encoder_decoder_layer_1(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(encoder_decoder_layer_1, self).__init__()
        if strides != 1:
            padding = "valid"
            n = 1
        else:
            padding = "same"
            n = 0
        self.padding = tf.keras.layers.ZeroPadding2D(padding=((n, 0), (n, 0)))
        self.Conv_DW = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                                       padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_1, training=None, **kwargs):
        x = self.padding(input_1)
        x = self.Conv_DW(x)
        x = self.bn(x)
        x = self.relu(x)
        print(x.shape)
        return x


class encoder_decoder_module(tf.keras.Model):
    def __init__(self):
        super(encoder_decoder_module, self).__init__()
        self.layer_1 = encoder_decoder_layer_1(filters=512, kernel_size=(3, 3), strides=2)
        self.layer_2 = encoder_decoder_layer_1(filters=256, kernel_size=(3, 3), strides=1)
        self.layer_3 = common.Conv_set(filters=128, kernel_size=(1, 1), strides=2)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, input, training=None, **kwargs):
        output_1 = input
        # input 52*52*256
        x = self.layer_1(input)
        # x 26*26*512
        x = self.layer_2(x)
        output_2 = x
        # x 26*26*256
        x = self.layer_1(x)
        # x 13*13*512
        x = self.layer_2(x)
        output_3 = x
        output_3 = self.layer_3(output_3)
        # x 13*13*256
        x = self.layer_2(x)
        # x 13*13*256
        x = self.upsampling(x)
        x = self.layer_2(x)
        output_2 = tf.add(output_2, x)
        x = self.layer_2(output_2)
        output_2 = self.layer_3(output_2)
        x = self.upsampling(x)
        x = self.layer_2(x)
        output_3 = tf.add(output_1, x)
        print(output_1.shape, output_2.shape, output_3.shape)
        return output_1, output_2, output_3
