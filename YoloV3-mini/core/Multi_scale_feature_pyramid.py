import common
import tensorflow as tf


def multi_scale_feature_pyramid(route_1, route_2, route_3):
    """
    this is multi-scale feature pyramid:
    include:
        1. concat module
        2. encoder-decoder module
        3. feature fusion module
    paremter:
        route_1: tensor from backbone, size: 13*13*1024
        route_2: tensor from backbone, size: 26*26*128
        route_3: tensor from backbone,size: 52*52*64
    """
    output_concat_1, output_concat_2, output_concat_3 = concat_module(route_1, route_2, route_3)
    output_encoder_1, output_encoder_2, output_encoder_3 = encoder_decoder_module(output_concat_1, output_concat_2,
                                                                                  output_concat_3)
    output_1, output_2, output3 = feature_fusion_module(route_1, route_2, route_3, output_encoder_1, output_encoder_2,
                                                        output_encoder_3)
    return output_1, output_2, output3


def concat_module(route_1, route_2, route_3,trainable):
    """
    this is concat module in multi-scale feature pyramid:
    input size:
        route_1: tensor from backbone, size: 13*13*1024
        route_2: tensor from backbone, size: 26*26*128
        route_3: tensor from backbone,size: 52*52*64
    """
    output_concat_1 = route_1
    temp_1 = common.conv2d(input_data=route_1, filters_shape=(1, 1, 1024, 512),trainable=trainable, name='conv1', activate=True, bn=True)
    temp_1 = common.upsample(temp_1, name='upsample1', method='resize')
    temp_2 = common.conv2d(input_data=route_2, filters_shape=(3, 3, 128, 256), trainable=trainable,name='conv2', activate=True, bn=True)
    temp_2 = tf.concat([temp_2, temp_1], axis=3)
    output_concat_2 = temp_2
    temp_2 = common.upsample(temp_2, name='upsample2', method='resize')
    temp_3 = common.conv2d(input_data=route_3, filters_shape=(3, 3, 64, 128), trainable=trainable,name='conv3', activate=True, bn=True)
    temp_3 = tf.concat([temp_3, temp_2], axis=3)
    temp_3 = common.conv2d(input_data=temp_3, filters_shape=(1, 1, 896, 256), trainable=trainable,name='conv4', activate=True, bn=True)
    output_concat_3 = temp_3
    return output_concat_1, output_concat_2, output_concat_3
