"""
this file include a class for upsampling, include:
1. nearest interpolation
2. bilinear interpolation
3. bicubic interpolation
4. transpose convolution
"""
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

    def nearest_interpolation(self):
        output = np.zeros((self.out_size[0], self.out_size[1]))
        for i in range(self.out_size[0]):
            for j in range(self.out_size[1]):
                new_i = round(i * self.input_size[0] / self.out_size[0])
                new_j = round(j * self.input_size[1] / self.out_size[1])
                if new_i>=self.input_size[0]:
                    new_i=self.input_size[0]-1
                if new_j>=self.input_size[1]:
                    new_j=self.input_size[1]-1
                output[i, j] = self.img[new_i, new_j]
        return output


def main():
    img = np.array([[1, 2],[5,6]])
    upsampling = Upsamlping(img, [5, 6])
    output = upsampling.nearest_interpolation()
    print(output)


if __name__ == '__main__':
    main()
