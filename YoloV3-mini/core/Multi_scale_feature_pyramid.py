import tensorflow as tf
from concat_module import concat_module
from encoder_decoder_module import encoder_decoder_module
from feature_fusion_module import feature_fusion
from predict_set import predict_set


class multi_scale_feature_pyramid(tf.keras.layers.Layer):
    def __init__(self,output_channels):
        super(multi_scale_feature_pyramid, self).__init__()
        self.concat = concat_module()
        self.encoder = encoder_decoder_module()
        self.feature_fusion_1 = feature_fusion(1152, 16)
        self.feature_fusion_2 = feature_fusion(896, 16)
        self.feature_fusion_3 = feature_fusion(384, 16)
        self.predict_set_1 = predict_set(1152, output_channels)
        self.predict_set_2 = predict_set(896, output_channels)
        self.predict_set_3 = predict_set(384, output_channels)

    def call(self, input_1, input_2, input_3, training=None, **kwargs):
        output_concat_1, output_concat_2, output_concat_3 = self.concat(input_1, input_2, input_3)
        output_encoder_1, output_encoder_2, output_encoder_3 = self.encoder(output_concat_3)
        output_feature_fusion_1 = self.feature_fusion_1(output_concat_1, output_encoder_1)
        output_feature_fusion_2 = self.feature_fusion_2(output_concat_2, output_encoder_2)
        output_feature_fusion_3 = self.feature_fusion_3(output_concat_3, output_encoder_3)
        final_output_1 = self.predict_set_1(output_feature_fusion_1)
        final_output_2 = self.predict_set_2(output_feature_fusion_2)
        final_output_3 = self.predict_set_3(output_feature_fusion_3)
        return final_output_1, final_output_2, final_output_3
