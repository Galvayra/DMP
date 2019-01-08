from DMP.learning.dataHandler import DataHandler
from DMP.learning.variables import GRAY_SCALE
from PIL import Image
import numpy as np
import math

slice56 = np.random.random((226, 226))

# print(slice56)
# # convert values to 0 - 255 int8 format
# formatted = (slice56 * 255 / np.max(slice56)).astype('uint8')
# img = Image.fromarray(formatted)
# img.show()


class ImageConverter:
    def __init__(self):
        self.dataHandler = DataHandler()

    def convert(self):

        x_train = self.dataHandler.x_train
        x_valid = self.dataHandler.x_valid
        x_test = self.dataHandler.x_test

        self.dataHandler.expand4square_matrix(*[x_train, x_valid, x_test])

        size = int(math.sqrt(len(x_train[0])))
        for data in x_train[:1]:
            data = np.array(data)
            data = np.reshape(data, (-1, size))
            img = Image.fromarray(data)
            # img.show()
