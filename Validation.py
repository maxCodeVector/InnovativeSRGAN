import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ImageFromFolder import ImageFrom2Folder

modelPath = 'checkpoint/model_epoch_56.pth'

def main(modelPath):
    G2 = torch.load(modelPath)['G2']
    img_data = ImageFrom2Folder('Dataset/ValidationTarget', 'Dataset/ValidationInput')
    data_loader = DataLoader(img_data, batch_size=2, shuffle=True)

    for i, (targetImage, inputImage) in enumerate(data_loader):
        inputImage = inputImage.cuda()
        outputImage = G2(inputImage)
        outputImage = outputImage.detach()
        draw(inputImage, outputImage, targetImage)




def draw(inputImage, outputImage, targetImage):
    inputImage = (inputImage.cpu().detach() + 1) * 122.5
    outputImage = (outputImage.cpu().detach() + 1) * 122.5
    targetImage = (targetImage + 1) * 122.5
    inputImage = inputImage[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    outputImage = outputImage[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    targetImage = targetImage[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

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