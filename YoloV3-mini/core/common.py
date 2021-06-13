import tensorflow as tf


class Conv_set(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, groups=1):
        super(Conv_set, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding="same", groups=groups)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, bn=True, activate=True, **kwargs):
        x = self.conv(inputs)
        if bn:
            x = self.bn(x, training=training)
        if activate:
            x = self.relu(x)
        return x


class DW_conv(tf.keras.layers.Layer):
    def __init__(self, kernel_size, strides):
        super(DW_conv, self).__init__()
        if strides != 1:
            padding = "valid"
            n=1
        else:
            padding = "same"
            n=0
        self.padding = tf.keras.layers.ZeroPadding2D(padding=((n,0),(n,0)))
        self.conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, bn=True, activate=True, **kwargs):
        x= self.padding(inputs)
        x = self.conv(x)
        if bn:
            x = self.bn(x, training=training)
        if activate:
            x = self.relu(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters1, filters2):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv_set(filters=filters1, kernel_size=(1, 1), strides=1, groups=4)
        self.conv2 = DW_conv(kernel_size=(3, 3), strides=1)
        self.conv3 = Conv_set(filters=filters2, kernel_size=(1, 1), strides=1, groups=4)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training, bn=True, activation=True)
        x = channel_shuffle(x)
        x = self.conv2(x, training=training, bn=True, activate=False)
        x = self.conv3(x, training=training, bn=False, activate=False)
        x = tf.keras.layers.add([x, inputs])
        x = self.relu(x)
        return x


def make_residual_block(filters1, filters2, num_blocks):
    x = tf.keras.Sequential()
    for _ in range(num_blocks):
        x.add(ResidualBlock(filters1=filters1, filters2=filters2))
    return x


class DW_seperable_conv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DW_seperable_conv, self).__init__()
        self.DW_conv = DW_conv(kernel_size=kernel_size,
                               strides=strides)
        self.PW_conv = Conv_set(filters=filters, kernel_size=(1, 1), strides=1)

    def call(self, inputs, training=None, bn=True, activate=True, **kwargs):
        x = self.DW_conv(inputs, training=training, bn=bn, activate=activate)
        x = self.PW_conv(x, training=training, bn=bn, activate=activate)
        return x


def channel_shuffle(input_data, group=4):
    channel_num = input_data.shape[-1]
    if channel_num % group != 0:
        raise ValueError("The group must be divisible by the shape of the last dimension of the input_data.")
    x = tf.reshape(input_data, shape=(-1, input_data.shape[1], input_data.shape[2], group, channel_num // group))
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    x = tf.reshape(x, shape=(-1, input_data.shape[1], input_data.shape[2], channel_num))
    return x


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.compat.v1.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize(input_data, (input_shape[1] * 2, input_shape[2] * 2), method="nearest")

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())

    return output
