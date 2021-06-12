import tensorflow as tf


def conv2d(input_data, filters_shape, trainable, name, downsample=False, activate=False, bn=False):
    """
    this is the traditional convolution
    input_data:[batch size, rows,cols, channels]
    ters_shape:[kernel_size,kernel_size,input_channel,output_channel]
    """
    with tf.compat.v1.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.Variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape,
                             initial_value=tf.random.normal(filters_shape, stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filters=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.keras.layers.BatchNormalization(inputs=conv, trainable=trainable)
        else:
            bias = tf.Variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                               initial_value=tf.zeros(filters_shape[-1]))
            conv = tf.nn.bias_add(conv, bias)

        if activate:
            conv = tf.nn.relu(conv)

    return conv


def conv_Depthwise_seperatable(input_data, filters_shape, trainable, name, downsample=False, activate=False, bn=False,
                               pointwise=True):
    """
    this is the depthwise seperable convolution, block in figure 4(e)
    """
    with tf.compat.v1.variable_scope(name):
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
        filters_shape_dw = (kernel_size, kernel_size, input_data.shape[-1], 1)
        weight = tf.Variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape_dw,
                             initial_value=tf.random.normal(filters_shape_dw, stddev=0.01))
        conv = tf.nn.depthwise_conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.keras.layers.BatchNormalization(inputs=conv, trainable=trainable)
        else:
            bias = tf.Variable(name='bias', shape=filters_shape_dw[-2], trainable=True, dtype=tf.float32,
                               initial_value=tf.zeros(filters_shape_dw[-2]))
            conv = tf.nn.bias_add(conv, bias)

        if activate:
            conv = tf.nn.relu(conv)
        # point-wise convoluvtion + BN + relu
        if pointwise:
            conv = convPW(conv, output_channels, trainable, name, activate=True, bn=True)

    return conv


def convPW(input_data, output_channels, trainable, name, activate=False, bn=False):
    """
    this is the pointwise convolution
    """
    with tf.compat.v1.variable_scope(name):
        strides = (1, 1, 1, 1)
        padding = "SAME"
        filters_shape = (1, 1, input_data.shape[-1], output_channels)
        weight = tf.Variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape,
                             initial_value=tf.random.normal(filters_shape, stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filters=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.keras.layers.BatchNormalization(inputs=conv, trainable=trainable)
        else:
            bias = tf.Variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                               initial_value=tf.zeros(filters_shape[-1]))
            conv = tf.nn.bias_add(conv, bias)

        if activate:
            conv = tf.nn.relu(conv)

    return conv


def conPW_group(input_data, output_channels, trainable, name, group=4, activate=False, bn=False):
    """
    this is the pointwise convolution with group
    """
    with tf.compat.v1.variable_scope(name):
        strides = (1, 1, 1, 1)
        padding = "SAME"
        filters_shape = (1, 1, int(input_data.shape[-1] / group), output_channels)

        weight = tf.Variable(name="weight", dtype=tf.float32, trainable=True, shape=filters_shape,
                             initial_value=tf.random.normal(filters_shape, stddev=0.01))
        input_groups = tf.split(value=input_data, num_or_size_splits=group, axis=3)
        weight_groups = tf.split(value=weight, num_or_size_splits=group, axis=3)
        groupConv = lambda i, k: tf.nn.conv2d(i, k, strides=strides, padding=padding)
        conv = [groupConv(i, k) for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(conv, axis=3)
        if bn:
            conv = tf.keras.layers.BatchNormalization(inputs=conv, trainable=trainable)
        else:
            bias = tf.Variable(name='bias', shape=filters_shape[-1], trainable=True, dtype=tf.float32,
                               initial_value=tf.zeros(filters_shape[-1]))
            conv = tf.nn.bias_add(conv, bias)

        if activate:
            conv = tf.nn.relu(conv)

    return conv


def channel_shuffle(input_data, group=4):
    channel_num = input_data.shape[-1]
    if channel_num % group != 0:
        raise ValueError("The group must be divisible by the shape of the last dimension of the input_data.")
    x = tf.reshape(input_data, shape=(-1, input_data.shape[1], input_data.shape[2], group, channel_num // group))
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    x = tf.reshape(x, shape=(-1, input_data.shape[1], input_data.shape[2], channel_num))
    return x


def residual_block(input_data, input_channel, filter_num1, filter_num2, filter_num3, trainable, name):
    """
    this is the network unit in figure 4(d) in paper
    """
    short_cut = input_data

    with tf.compat.v1.variable_scope(name):
        input_data = conPW_group(input_data, filter_num1, trainable=trainable, name='conv1', group=4, activate=True,
                                 bn=True)
        input_data = channel_shuffle(input_data)
        input_data = conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, input_channel, filter_num2),
                                                trainable=trainable, name='conv2', bn=True, activate=False,
                                                pointwise=False)
        input_data = conPW_group(input_data, filter_num3, trainable=trainable, name='conv3', group=4, activate=False,
                                 bn=False)
        residual_output = input_data + short_cut
        residual_output = tf.nn.relu(residual_output)

    return residual_output


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
