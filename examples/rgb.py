import math

import numpy

import ndv

img = numpy.zeros((4, 256, 256), dtype=numpy.uint8)

for x in range(256):
    for y in range(256):
        img[0, x, y] = x
        img[1, x, y] = y
        img[2, x, y] = 255 - x
        img[3, x, y] = int(math.sqrt((x - 128) ** 2 + (y - 128) ** 2))

n = ndv.imshow(img)
