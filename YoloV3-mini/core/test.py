import tensorflow as tf

tf.compat.v1.disable_eager_execution()
import backbone


def main():
    input_data = tf.ones([1, 416, 416, 3])
    output1, output2, output3 = backbone.backbone(input_data, True)
    print(output1.shape)
    print(output2.shape)
    print(output3.shape)


if __name__ == '__main__':
    main()
