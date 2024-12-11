import math

import numpy

import ndv

img = numpy.zeros((256, 256, 4), dtype=numpy.uint8)

for x in range(256):
    for y in range(256):
        img[x, y, 0] = x
        img[x, y, 1] = y
        img[x, y, 2] = 255 - x
        img[x, y, 3] = int(math.sqrt((x - 128) ** 2 + (y - 128) ** 2))

n = ndv.imshow(img)
