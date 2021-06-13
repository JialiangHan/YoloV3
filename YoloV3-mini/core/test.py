import tensorflow as tf
import backbone


# import common
# from Multi_scale_feature_pyramid import concat_module


def main():
    input_data = tf.ones([1, 416, 416, 3])
    # output = tf.keras.layers.Conv2D(filters=3,
    #                                        kernel_size=(3,3),
    #                                        strides=2,
    #                                        padding="valid")(input_data)
    # print(output.shape)
    b1 = backbone.backbone()
    output1, output2, output3 = b1(input_data, training=True)
    # output_concat_1, output_concat_2, output_concat_3 = concat_module(output1, output2, output3, True)


if __name__ == '__main__':
    main()
