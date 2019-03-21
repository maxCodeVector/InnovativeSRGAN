import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ImageFromFolder import ImageFrom2Folder
from skimage import measure
import main2DD

modelPath = 'checkPointOY/2DD_120.pth'
psnr = []
ssim = []


def main(model):
    # model = main2DD.getModel(modelPath)
    G2 = model['G2']
    G1 = model['G1']
    G2.eval()
    G1.eval()
    img_data = ImageFrom2Folder('Dataset/ValidationTarget', 'Dataset/ValidationInput', num=10)
    data_loader = DataLoader(img_data, batch_size=1, shuffle=True)
    for i, (targetImage, inputImage) in enumerate(data_loader):
        inputImage = inputImage.cuda()
        outputImage = G2(inputImage).detach()
        draw(inputImage.cpu(), outputImage.cpu(), targetImage.cpu())


def draw(inputImage, outputImage, targetImage):
    inputImage = (inputImage + 1) * 122.5
    outputImage = (outputImage + 1) * 122.5
    targetImage = (targetImage + 1) * 122.5
    inputImage = inputImage[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    outputImage = outputImage[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    targetImage = targetImage[0].numpy().astype(np.uint8).transpose(1, 2, 0)

    ImagePSNR = measure.compare_psnr(outputImage, targetImage)
    ImageSSIM = measure.compare_ssim(outputImage, targetImage, multichannel=True)

    plt.figure(figsize=(12, 4))
    ax = plt.subplot("131")
    ax.imshow(inputImage)
    ax.set_title("inputImage")

    ax = plt.subplot("132")
    ax.imshow(outputImage)
    ax.set_title("outputImage")

    ax = plt.subplot("133")
    ax.imshow(targetImage)
    ax.set_title("GroundTruth")

    plt.figtext(0.09, 0, "PSNR: %s" % ImagePSNR)
    plt.figtext(0.76, 0, "SSIM: %s" % ImageSSIM)
    plt.show()

    psnr.append(ImagePSNR)
    ssim.append(ImageSSIM)


if __name__ == "__main__":
    main(modelPath)

    print('psnr average:', np.array(psnr).mean())
    print('ssim average:', np.array(ssim).mean())
    fig = plt.figure()
    ax1 = fig.add_subplot('111')
    ax2 = ax1.twinx()
    ax1.plot(list(range(len(psnr))), psnr, 'r+', label='PSNR')
    ax2.plot(list(range(len(ssim))), ssim, 'g*', label='SSIM')
    ax1.set_ylabel('PSNR value')
    ax1.set_xlabel('picture index')
    ax2.set_ylabel('SSIM value')
    ax1.legend(loc='upper right')
    plt.grid(True)
    plt.show()