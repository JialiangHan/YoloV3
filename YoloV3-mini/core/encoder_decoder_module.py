import tensorflow as tf


class encoder_decoder_layer_1(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(encoder_decoder_layer_1, self).__init__()
        self.input_spec = tf.keras.layers.InputSpec(
            shape=(None, None, None, 256))
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
        # print(x.shape)
        return x


class encoder_decoder_layer_2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(encoder_decoder_layer_2, self).__init__()
        self.input_spec = tf.keras.layers.InputSpec(
            shape=(None, None, None, 512))
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
        # print(x.shape)
        return x


class encoder_decoder_layer_3(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(encoder_decoder_layer_3, self).__init__()
        self.input_spec = tf.keras.layers.InputSpec(
            shape=(None, None, None, 256))
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
        # print(x.shape)
        return x


class encoder_decoder_layer_4(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(encoder_decoder_layer_4, self).__init__()
        self.input_spec = tf.keras.layers.InputSpec(
            shape=(None, None, None, 256))

        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding='valid')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.strides = strides

    def call(self, input, training=None, **kwargs):
        input_h = input.shape[2]
        n = int((self.strides*input_h-input_h)/2)
        x = tf.keras.layers.ZeroPadding2D(padding=((n, n), (n, n)))(input)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.shape)
        return x


class encoder_decoder_module(tf.keras.layers.Layer):
    def __init__(self):
        super(encoder_decoder_module, self).__init__()
        self.layer_1 = encoder_decoder_layer_1(filters=512, kernel_size=(3, 3), strides=2)
        self.layer_2 = encoder_decoder_layer_2(filters=256, kernel_size=(3, 3), strides=1)
        self.layer_3 = encoder_decoder_layer_3(filters=256, kernel_size=(3, 3), strides=1)
        self.layer_4 = encoder_decoder_layer_4(filters=128, kernel_size=(1, 1), strides=2)
        self.upsampling = tf.keras.layers.UpSampling2D(size=(2, 2))

    def call(self, input, training=None, **kwargs):
        output_1 = input
        x = self.layer_1(input)
        x = self.layer_2(x)
        # print(x)
        output_2 = x
        x = self.layer_1(x)
        x = self.layer_2(x)
        output_3 = self.layer_4(x)
        x = self.layer_3(x)
        x = self.upsampling(x)
        x = self.layer_3(x)
        output_2 = tf.add(output_2, x)
        x = self.layer_3(output_2)
        output_2 = self.layer_4(output_2)
        x = self.upsampling(x)
        x = self.layer_3(x)
        output_1 = tf.add(output_1, x)
        output_1 = self.layer_4(output_1)
        # print(output_3.shape, output_2.shape, output_1.shape)
        return output_3, output_2, output_1
