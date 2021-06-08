import tensorflow as tf


def conv2d(input_data, filters_shape, trainable, name, downsample=False, activate=False, bn=False):
    """
    this is the traditional convolution
    """
    with tf.varable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.get_variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.keras.layers.batch_normalization(conv, training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(bias)

        if activate:
            conv = tf.nn.relu(conv)

    return conv


def conv_Depthwise_seperatable(input_data, filters_shape, trainable, name, downsample=False, activate=False, bn=False):
    """
    this is the depthwise seperable convolution, block in figure 4(e)
    """
    with tf.varable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"
        output_channels = filters_shape[-1]
        kernel_size = filters_shape[0]
        filters_shape_dw = (kernel_size, kernel_size, input_data[-1], 1)
        weight = tf.get_variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape_dw,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.depthwise_conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.keras.layers.batch_normalization(conv, training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(bias)

        if activate:
            conv = tf.nn.relu(conv)
        # point-wise convoluvtion + BN + relu
        conv = convPW(conv, output_channels, trainable, name, activate=True, bn=True)

    return conv


def convPW(input_data, output_channels, trainable, name, activate=False, bn=False):
    """
    this is the pointwise convolution
    """
    with tf.varable_scope(name):
        strides = (1, 1, 1, 1)
        padding = "SAME"
        filters_shape = (1, 1, input_data[-1], output_channels)
        weight = tf.get_variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.keras.layers.batch_normalization(conv, training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(bias)

        if activate:
            conv = tf.nn.relu(conv)

    return conv


def resisual_block():
    """
    this is the network unit in figure 4(d) in paper
    """
