import common
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


def backbone(input_data, trainable):
    with tf.variable_scope('backbone'):

        input_data = common.conv2d(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0')
        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 32, 16),
                                                       trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 16, 16, 16, 16, trainable=trainable,
                                               name='residual%d' % (i + 0))

        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 16, 32),
                                                       trainable=trainable, name='conv5', downsample=True)

        for i in range(2):
            input_data = common.residual_block(input_data, 32, 128, 128,32, trainable=trainable,
                                               name='residual%d' % (i + 1))

        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 32, 64),
                                                       trainable=trainable, name='conv12', downsample=True)

        for i in range(6):
            input_data = common.residual_block(input_data, 64, 256, 256,64, trainable=trainable,
                                               name='residual%d' % (i + 3))

        route_1 = input_data
        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 64, 128),
                                                       trainable=trainable, name='conv31', downsample=True)

        for i in range(6):
            input_data = common.residual_block(input_data,128, 512, 512, 128, trainable=trainable,
                                               name='residual%d' % (i + 9))

        route_2 = input_data
        input_data = common.conv_Depthwise_seperatable(input_data, filters_shape=(3, 3, 128, 256),
                                                       trainable=trainable, name='conv50', downsample=True)

        for i in range(1):
            input_data = common.residual_block(input_data, 256,1024, 1024, 256, trainable=trainable,
                                               name='residual%d' % (i + 15))
        input_data = common.conv2d(input_data, filters_shape=(1, 1, 256, 1024), trainable=trainable, name='conv54')

        return route_1, route_2, input_data
