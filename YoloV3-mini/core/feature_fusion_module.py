"""
this is the feature fusion module:
input:
1. output from cancate module
    1. 13*13*1024
    2. 26*26*768
    3. 52*52*256
2. output from encoder decoder module
    2.1 13*13*128
    2.2 26*26*128
    2.3 52*52*128

"""
import tensorflow as tf


class feature_fusion(tf.keras.layers.Layer):
    def __init__(self, output_channels, reduction_ratio):
        super(feature_fusion, self).__init__()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.fc_1 = tf.keras.layers.Dense(units=output_channels/reduction_ratio,activation="relu")
        self.fc_2 = tf.keras.layers.Dense(units=output_channels,activation='sigmoid')

    def call(self, input_1,input_2, **kwargs):
        input = self.concat([input_1,input_2])
        squeeze = self.global_avg_pooling(input)
        excitation = self.fc_1(squeeze)
        excitation = self.fc_2(excitation)
        excitation = tf.reshape(excitation,[-1,1,1,input.shape[-1]])
        output = input*excitation
        return output

