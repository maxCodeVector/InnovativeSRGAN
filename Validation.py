import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ImageFromFolder import ImageFrom2Folder

modelPath = 'checkpoint/model_epoch_90.pth'

def main(modelPath):
    G2 = torch.load(modelPath)['G2']
    G1 = torch.load(modelPath)['G1']
    img_data = ImageFrom2Folder('../Dataset/ValidationTarget', '../Dataset/ValidationInput')
    data_loader = DataLoader(img_data, batch_size=1, shuffle=True)
    G2.eval()
    G1.eval()
    for i, (targetImage, inputImage) in enumerate(data_loader):
        inputImage = inputImage.cuda()
        outputImage = G2(inputImage).detach()
        # outputImage = outputImage.detach()
        output1 = G1(inputImage).detach()
        draw(inputImage.cpu(), outputImage.cpu(), targetImage.cpu())
        # draw(inputImage.cpu(), output1.cpu(), targetImage.cpu())




def draw(inputImage, outputImage, targetImage):
    inputImage = (inputImage + 1) * 122.5
    outputImage = (outputImage + 1) * 122.5
    targetImage = (targetImage + 1) * 122.5
    inputImage = inputImage[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    outputImage = outputImage[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    targetImage = targetImage[0].numpy().astype(np.uint8).transpose(1, 2, 0)

    plt.figure()
    ax = plt.subplot("131")
    ax.imshow(inputImage)
    ax.set_title("inputImage")

    # plt.figure()
    ax = plt.subplot("132")
    ax.imshow(outputImage)
    ax.set_title("outputImage")

    # plt.figure()
    ax = plt.subplot("133")
    ax.imshow(targetImage)
    ax.set_title("targetImage")

    plt.show()

if __name__ == "__main__":
    main(modelPath)