import tensorflow as tf


class predict_set(tf.keras.layers.Layer):
    def __init__(self, filter_1, filter_2):
        super(predict_set, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=filter_1 / 2,
                                             kernel_size=(1, 1),
                                             strides=1, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters=filter_1,
                                             kernel_size=(3, 3),
                                             strides=1, padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters=filter_2,
                                             kernel_size=(1, 1),
                                             strides=1, padding='same')

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x
