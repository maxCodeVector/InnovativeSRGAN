from PIL import Image
import scipy.misc as misc
import random
import numpy as np
import cv2 as cv


# 采样函数定义
def downscale(name, scale, output_dir=''):
    print(name)
    with Image.open(name) as im:
        w, h = im.size
        w_new = int(w / scale)
        h_new = int(h / scale)
        im_new = im.resize((w_new, h_new), Image.ANTIALIAS)
        if output_dir:
            import os
            save_name = os.path.join(output_dir, name.split(r'\\')[-1])
            im_new.save(save_name)


def resize(image, newsize):
    # methods = ['nearest','lanczos','bilinear','bicubic','cubic']
    methods = ['bicubic', 'lanczos', 'bilinear', 'bicubic', 'cubic']
    scale = [4, 3, 2, 1.5, 0.75, 0.5, 0.25]
    # 0.309090465205
    value = random.random()
    if (len(image.shape) == 3):
        w, h, d = image.shape
    else:
        w, h = image.shape
    s = int(value * 1000) % 7
    if s == 0:
        image = image + np.random.random_sample(image.shape) * (int(value * 100) % 50)
    elif s == 1:
        image = cv.blur(image, (5, 5))
    elif s == 2:
        image = cv.GaussianBlur(image, (5, 5), 1)
    elif s == 3:
        image = cv.medianBlur(image, 5)
    elif s == 4:
        cv.bilateralFilter(image, 9, 75, 75)

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
