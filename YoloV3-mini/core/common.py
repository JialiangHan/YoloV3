import tensorflow as tf


def conv2d(input_data, filters_shape, trainable, name, activate=False, bn=False):
    """
    this is the traditional convolution
    """
    with tf.varable_scope(name):
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


def convDW():
    """
    this is the depthwise convolution
    """

def convPW():
    """
    this is the pointwise convolution
    """

def resisual_block():
    """
    this is the network unit in figure 4(d) in paper
    """
