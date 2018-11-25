import numpy as np
import os

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import downsimple


extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
transformHR = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])
transformLR = transforms.Compose([
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])


class ImageFrom2Folder(data.Dataset):
    def __init__(self, HRpath, LRpath, num=10):
        super(ImageFrom2Folder, self).__init__()
        # self.HRimages = make_dataset(HRpath)
        # self.LRimages = make_dataset(LRpath)
        HRimages = make_dataset(HRpath)
        LRimages = make_dataset(LRpath)
        self.HRimages = []
        self.LRimages = []
        for HR in HRimages[0:num]:
            self.HRimages.append(pil_loader(HR, True))
        for LR in LRimages[0:num]:
            self.LRimages.append(pil_loader(LR, False))

    def __getitem__(self, index):
        # HR = pil_loader(self.HRimages[index], True)
        # LR = pil_loader(self.LRimages[index], False)
        # return (HR, LR)
        #
        return self.HRimages[index], self.LRimages[index]

    def __len__(self):
        return len(self.HRimages)


# class ImageFrom1Folder(data.Dataset):
#     def __init__(self, path):
#         super(ImageFrom1Folder, self).__init__()
#         self.images = make_dataset(path)
#
#     def __getitem__(self, index):
#         image = pil_loader(self.images[index], False)
#         return image
#
#     def __len__(self):
#         return len(self.images)


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)

    return images


def has_file_allowed_extension(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def pil_loader(path, HR):  # 根据地址读取图像
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
        if HR:
            img = transformHR(img)
        else:
            img = transformLR(img)
        return img

# i = ImageFromFolder('Dataset/HRimages', 'Dataset/LRimages')
# print(i.__getitem__(0)[0].size())
# i.__getitem__(0)[1].show()


def get_downsimple_Tensor(images):
    res = []
    high = ((images + 1) * 255 / 2).numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    for img in high:
        dows = downsimple.newDownscale(img, 4)
        dows = transforms.ToTensor()(np.array(dows))
        dows = transforms.Normalize((.5, .5, .5), (.5, .5, .5))(dows)
        res.append(dows.numpy())
    return torch.Tensor(res)
