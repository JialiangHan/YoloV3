import tensorflow as tf
import backbone


# import common
from concat_module import concat_module


def main():
    input_data = tf.ones([1, 416, 416, 3])
    b1 = backbone.backbone()
    output1, output2, output3 = b1(input_data, training=True)
    c = concat_module()
    output_concat_1, output_concat_2, output_concat_3 = c(output1, output2, output3)


if __name__ == '__main__':
    main()
