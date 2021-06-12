import common
import tensorflow as tf


def backbone(input_data, trainable):
    with tf.compat.v1.variable_scope('backbone'):

        input_data = common.conv2d(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv1')
        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 32, 16),
                                                       trainable=trainable, name='conv2', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 16, 16, 16, 16, trainable=trainable,
                                               name='residual%d' % (i + 1))

        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 16, 32),
                                                       trainable=trainable, name='conv6', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 32, 128, 128,32, trainable=trainable,
                                               name='residual%d' % (i + 2))

        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 32, 64),
                                                       trainable=trainable, name='conv13', downsample=True)

        for i in range(6):
            input_data = common.residual_block(input_data, 64, 256, 256,64, trainable=trainable,
                                               name='residual%d' % (i + 4))

        route_3 = input_data
        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 64, 128),
                                                       trainable=trainable, name='conv32', downsample=True)

        for i in range(6):
            input_data = common.residual_block(input_data,128, 512, 512, 128, trainable=trainable,
                                               name='residual%d' % (i + 10))

        route_2 = input_data
        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 128, 256),
                                                       trainable=trainable, name='conv51', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 256,1024, 1024, 256, trainable=trainable,
                                               name='residual%d' % (i + 16))
        input_data = common.conv2d(input_data, filters_shape=(1, 1, 256, 1024), trainable=trainable, name='conv55')
        route_1 = input_data
        return route_1, route_2, route_3
