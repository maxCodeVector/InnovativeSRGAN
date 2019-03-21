import os
from PIL import Image
from torch.utils import data
from torchvision import transforms


extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
transformHR = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transformLR = transforms.Compose([
    transforms.CenterCrop(64),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ImageFrom2Folder(data.Dataset):
    def __init__(self, HRpath, LRpath, num=-1):
        super(ImageFrom2Folder, self).__init__()
        HRimages = make_dataset(HRpath)
        LRimages = make_dataset(LRpath)
        self.HRimages = []
        self.LRimages = []
        for HR in HRimages[0:num if num >= 0 else len(HRimages)]:
            self.HRimages.append(pil_loader(HR, True))
        for LR in LRimages[0:num if num >= 0 else len(LRimages)]:
            self.LRimages.append(pil_loader(LR, False))

    def __getitem__(self, index):
        return self.HRimages[index], self.LRimages[index]

    def __len__(self):
        return len(self.HRimages)


class ImageFrom3Folder(data.Dataset):
    def __init__(self, HRpath, LRpath, referencePath, num=-1):
        super(ImageFrom3Folder, self).__init__()
        HRimages = make_dataset(HRpath)
        LRimages = make_dataset(LRpath)
        referenceImages = make_dataset(referencePath)
        self.HRimages = []
        self.LRimages = []
        self.referenceImages = []
        for HR in HRimages[0:num if num >= 0 else len(HRimages)]:
            self.HRimages.append(pil_loader(HR, True))
        for LR in LRimages[0:num if num >= 0 else len(LRimages)]:
            self.LRimages.append(pil_loader(LR, False))
        for RI in referenceImages[0:num if num >= 0 else len(referenceImages)]:
            self.referenceImages.append(pil_loader(RI, True))

    def __getitem__(self, index):
        return self.HRimages[index], self.LRimages[index], self.referenceImages[index]

    def __len__(self):
        return len(self.HRimages)


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
