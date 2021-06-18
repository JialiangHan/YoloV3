import tensorflow as tf
from YoloV3_mini import YoloV3_mini


def main():
    input_data = tf.ones([1, 416, 416, 3])
    y = YoloV3_mini(255)
    y.build(input_shape=[1, 416, 416, 3])
    y.summary()
    final_output_1, final_output_2, final_output_3 = y(input_data)

    print(final_output_1.shape)
    print(final_output_2.shape)
    print(final_output_3.shape)


if __name__ == '__main__':
    main()
