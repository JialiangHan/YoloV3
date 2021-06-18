import tensorflow as tf
import backbone
from concat_module import concat_module
from encoder_decoder_module import encoder_decoder_module
from feature_fusion_module import feature_fusion
from predict_set import predict_set

def main():
    input_data = tf.ones([1, 416, 416, 3])
    b1 = backbone.backbone()
    output1, output2, output3 = b1(input_data, training=True)
    c = concat_module()
    output_concat_1, output_concat_2, output_concat_3 = c(output1, output2, output3)
    e = encoder_decoder_module()
    output_encoder_1, output_encoder_2, output_encoder_3 = e(output_concat_3)
    f1 = feature_fusion(1152, 16)
    output_feature_fusion_1 = f1(output_concat_1, output_encoder_1)
    f2 = feature_fusion(896, 16)
    output_feature_fusion_2 = f2(output_concat_2, output_encoder_2)
    f3 = feature_fusion(384, 16)
    output_feature_fusion_3 = f3(output_concat_3, output_encoder_3)
    p1 = predict_set(output_feature_fusion_1.shape[-1],255)
    p2 = predict_set(output_feature_fusion_2.shape[-1],255)
    p3 = predict_set(output_feature_fusion_3.shape[-1],255)
    final_output_1 = p1(output_feature_fusion_1)
    final_output_2 = p2(output_feature_fusion_2)
    final_output_3 = p3(output_feature_fusion_3)
    print(final_output_1.shape)
    print(final_output_2.shape)
    print(final_output_3.shape)
    


if __name__ == '__main__':
    main()
