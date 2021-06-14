import common
import tensorflow as tf


class backbone(tf.keras.Model):
    def __init__(self):
        super(backbone, self).__init__()
        self.conv1 = common.Conv_set(filters=32, kernel_size=(3, 3), strides=1)
        self.conv2 = common.DW_seperable_conv(filters=16, kernel_size=(3, 3), strides=2)
        self.residual1 = common.make_residual_block(filters1=16, filters2=16, num_blocks=1)
        self.conv3 = common.DW_seperable_conv(filters=32, kernel_size=(3, 3), strides=2)
        self.residual2 = common.make_residual_block(filters1=128, filters2=32, num_blocks=2)
        self.conv4 = common.DW_seperable_conv(filters=64, kernel_size=(3, 3), strides=2)
        self.residual3 = common.make_residual_block(filters1=256, filters2=64, num_blocks=6)
        self.conv5 = common.DW_seperable_conv(filters=128, kernel_size=(3, 3), strides=2)
        self.residual4 = common.make_residual_block(filters1=512, filters2=128, num_blocks=6)
        self.conv6 = common.DW_seperable_conv(filters=256, kernel_size=(3, 3), strides=2)
        self.residual5 = common.make_residual_block(filters1=1024, filters2=256, num_blocks=1)
        self.conv7 = common.Conv_set(filters=1024, kernel_size=(1, 1), strides=1)

    def call(self, inputs, training=None, bn=True, activate=True, **kwargs):
        x = self.conv1(inputs, training=None, bn=False, activate=False)
        x = self.conv2(x, training=None, bn=True, activate=True)
        x = self.residual1(x, training=training)
        x = self.conv3(x, training=None, bn=True, activate=True)
        x = self.residual2(x, training=training)
        x = self.conv4(x, training=None, bn=True, activate=True)
        output_1 = self.residual3(x, training=training)
        x = self.conv5(output_1, training=None, bn=True, activate=True)
        output_2 = self.residual4(x, training=training)
        x = self.conv6(output_2, training=None, bn=True, activate=True)
        x = self.residual5(x, training=training)
        output_3 = self.conv7(x, training=None, bn=False, activate=False)
        # print(output_3.shape, output_2.shape, output_1.shape)
        return output_3, output_2, output_1
