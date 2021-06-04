"""
this file include a class for upsampling, include:
1. nearest interpolation
2. bilinear interpolation

"""
import math

import numpy as np


class Upsamlping:
    def __init__(self, input_img, output_size):
        """
        input_size=[  width, height]
        output size=[ width, height]
        """
        self.img = input_img
        self.input_size = np.shape(self.img)
        self.out_size = output_size
        self.output = np.zeros((self.out_size[0], self.out_size[1]))

    def nearest_interpolation(self):
        for i in range(self.out_size[0]):
            for j in range(self.out_size[1]):
                new_i = round(i * self.input_size[0] / self.out_size[0])
                new_j = round(j * self.input_size[1] / self.out_size[1])
                if new_i >= self.input_size[0]:
                    new_i = self.input_size[0] - 1
                if new_j >= self.input_size[1]:
                    new_j = self.input_size[1] - 1
                self.output[i, j] = self.img[new_i, new_j]
        return self.output

    def bilinear_interpolation(self):
        for i in range(self.out_size[0]):
            for j in range(self.out_size[1]):
                # center_i = (i + 0.5) * self.input_size[0] / self.out_size[0] - 0.5
                # center_j = (j + 0.5) * self.input_size[1] / self.out_size[1] - 0.5
                center_i = (i ) * self.input_size[0] / self.out_size[0]
                center_j = (j ) * self.input_size[1] / self.out_size[1]
                left_up_x = math.floor(center_i)
                left_up_y = math.floor(center_j)
                right_down_x = min(left_up_x + 1, self.input_size[0] - 1)
                right_down_y = min(left_up_y + 1, self.input_size[1] - 1)

                f1 = (right_down_y - center_j) * self.img[left_up_x, left_up_y] \
                     + (center_j - left_up_y) * self.img[right_down_x, left_up_y]
                f2 = (right_down_y - center_j) * self.img[left_up_x, right_down_y] \
                     + (center_j - left_up_y) * self.img[right_down_x, right_down_y]
                self.output[i, j] = (left_up_x - center_i) * f1 + (center_i - right_down_x) * f2
        return self.output


def main():
    img = np.array([[1, 2], [5, 6]])
    upsampling = Upsamlping(img, [4, 4])
    # output = upsampling.nearest_interpolation()
    output = upsampling.bilinear_interpolation()
    print(output)


if __name__ == '__main__':
    main()
