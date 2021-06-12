import tensorflow as tf
import backbone
from Multi_scale_feature_pyramid import concat_module


def main():
    input_data = tf.ones([1, 416, 416, 3])
    output1, output2, output3 = backbone.backbone(input_data, True)
    output_concat_1, output_concat_2, output_concat_3 = concat_module(output1, output2, output3, True)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)
    print(output_concat_1.shape)
    print(output_concat_2.shape)
    print(output_concat_3.shape)


if __name__ == '__main__':
    main()
