import torch
import scipy.misc as misc
import random
import numpy as np
# import cv2 as cv
# from matplotlib import transforms
from torchvision import transforms


def resize(image, newsize):
    methods = ['bicubic', 'lanczos', 'bilinear', 'bicubic', 'cubic']
    scale = [4, 3, 2, 1.5, 0.75, 0.5, 0.25]
    value = 0
    if (len(image.shape) == 3):
        w, h, d = image.shape
    else:
        w, h = image.shape
    # s = int(value * 1000) % 7
    # if s == 0:
    #     image = image + np.random.random_sample(image.shape) * (int(value * 100) % 50)
    # elif s == 1:
    #     image = cv.blur(image, (5, 5))
    # elif s == 2:
    #     image = cv.GaussianBlur(image, (5, 5), 1)
    # elif s == 3:
    #     image = cv.medianBlur(image, 5)
    # elif s == 4:
    #     cv.bilateralFilter(image, 9, 75, 75)

    image = misc.imresize(image, newsize, methods[int(value * 10) % 5])
    for i in range(int(value * 100) % 5):
        image = misc.imresize(image, (int(newsize[0] * scale[int(value * (10 ** (1 + i))) % 7]),
                                      int(newsize[1] * scale[int(value * (10 ** (1 + i))) % 7])),
                              methods[int(value * (10 ** (4 + i)) % 5)])
        image = misc.imresize(image, newsize, methods[int(value * (10 ** (4 + i)) % 10 - 5)])
    return image


def newDownscale(image, scale):
    w, h, d = image.shape
    w_new = int(w / scale)
    h_new = int(h / scale)
    image = resize(image, (w_new, h_new))
    return image


def get_downsimple_Tensor(images):
    res = []
    high = ((images + 1) * 255 / 2).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    for img in high:
        dows = newDownscale(img, 4)
        dows = transforms.ToTensor()(np.array(dows))
        dows = transforms.Normalize((.5, .5, .5), (.5, .5, .5))(dows)
        res.append(dows.numpy())
    return torch.Tensor(res)

