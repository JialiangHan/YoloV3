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



