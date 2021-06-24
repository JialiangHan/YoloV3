import tensorflow as tf
from backbone import backbone
from Multi_scale_feature_pyramid import multi_scale_feature_pyramid


class YoloV3_mini(tf.keras.Model):
    def __init__(self, output_channels):
        super(YoloV3_mini, self).__init__()
        self.b = backbone()
        self.msfp = multi_scale_feature_pyramid(output_channels)

    def call(self, inputs, training=None, **kwargs):
        output_b_1, output_b_2, output_b_3 = self.b(inputs, training=training)
        final_output_1, final_output_2, final_output_3 = self.msfp(output_b_1, output_b_2, output_b_3)
        return [final_output_1, final_output_2, final_output_3]


def yolov3_mini(inputs, output_channels):
    b = backbone()
    output_b_1, output_b_2, output_b_3 = b(inputs)
    msfp = multi_scale_feature_pyramid(output_channels)
    final_output_1, final_output_2, final_output_3 = msfp(output_b_1, output_b_2, output_b_3)
    return [final_output_3, final_output_2, final_output_1]
