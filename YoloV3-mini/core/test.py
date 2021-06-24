import tensorflow as tf
from YoloV3_mini import YoloV3_mini,yolov3_mini


def main():
    input_data = tf.ones([16, 416, 416, 3])
    # y = YoloV3_mini(255)
    # y.build(input_shape=[1, 416, 416, 3])2,
    # y.summary()
    # final_output = y(input_data)
    final_output = yolov3_mini(input_data,255)

    print(final_output[0].shape)
    print(final_output[1].shape)
    print(final_output[2].shape)


if __name__ == '__main__':
    main()
