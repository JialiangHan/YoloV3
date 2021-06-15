import tensorflow as tf
import backbone
from concat_module import concat_module
from encoder_decoder_module import encoder_decoder_module


def main():
    input_data = tf.ones([1, 416, 416, 3])
    b1 = backbone.backbone()
    output1, output2, output3 = b1(input_data, training=True)
    c = concat_module()
    output_concat_1, output_concat_2, output_concat_3 = c(output1, output2, output3)
    # output_concat_3 = tf.ones([1,52,52,256])
    # out = tf.keras.layers.SeparableConv2D(filters=256,kernel_size=(3,3))(output_concat_3)
    # print(out.shape)
    e = encoder_decoder_module()
    output_encoder_1, output_encoder_2, output_encoder_3 = e(output_concat_3)


if __name__ == '__main__':
    main()
