import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ImageFromFolder import ImageFrom2Folder
from model import *

num_epoch = 100
ResumePath = './checkpoint/model_epoch_11.pth'  # 输入resume的path,''为从头开始
initlr = 0.0001
start_epoch = 1
steps = 10

def Label(shape, isReal):
    real = torch.Tensor([1]).cuda()
    fake = torch.Tensor([0]).cuda()
    if isReal:
        return real.expand_as(shape)
    else:
        return fake.expand_as(shape)

# def draw(title, img):
#     img = img[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
#     plt.figure()
#     plt.imshow(img)
#     plt.title(title)
#     plt.show()


def draw(inputHR, G1output, G2output, targetLR):
    inputHR = (inputHR.cpu() + 1) * 122.5
    G1output = (G1output.cpu().detach() + 1) * 122.5
    G2output = (G2output.cpu().detach() + 1) * 122.5
    targetLR = (targetLR.cpu() + 1) * 122.5
    inputHR = inputHR[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    G1output = G1output[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    G2output = G2output[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    targetLR = targetLR[0].numpy().astype(np.uint8).transpose(1, 2, 0)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot("221")
    ax.imshow(inputHR)
    ax.set_title("inputHR")

    # plt.figure()
    ax = plt.subplot("222")
    ax.imshow(G1output)
    ax.set_title("G1output")

    # plt.figure()
    ax = plt.subplot("223")
    ax.imshow(G2output)
    ax.set_title("G2output")

    ax = plt.subplot("224")
    ax.imshow(targetLR)
    ax.set_title("targetLR")

    plt.show()


def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = initlr * (0.1 ** (epoch / steps))
    return lr


def main(resumePath):
    dataset_path = r'C:\Users\hya\Downloads\pytorch-SRResNet-master\pytorch-SRResNet-master\data\%d'
    # img_data = ImageFrom2Folder('Dataset/HRimages', 'Dataset/LRimages')
    img_data = ImageFrom2Folder(dataset_path % 20, dataset_path % 20)
    data_loader = DataLoader(img_data, batch_size=2, shuffle=True)

    G1 = Net_G()
    G2 = Reverse_G()
    D = Net_D()

    creterion = nn.MSELoss()
    optimizer_G = optim.Adam(itertools.chain(
                G1.parameters(), G2.parameters()), lr=initlr)
    optimizer_D = optim.Adam(D.parameters(), lr=initlr)

    if torch.cuda.is_available():
        D = D.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()

    if resumePath:
        if os.path.isfile(resumePath):
            print("=> loading checkpoint '{}'".format(resumePath))
            checkpoint = torch.load(resumePath)
            start_epoch = checkpoint["epoch"] + 1
            D.load_state_dict(checkpoint["D"].state_dict())
            G1.load_state_dict(checkpoint["G1"].state_dict())
            G2.load_state_dict(checkpoint["G2"].state_dict())
            optimizer_G.load_state_dict(checkpoint["optimizerG"].state_dict())
            optimizer_D.load_state_dict(checkpoint["optimizerD"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(resumePath))
            start_epoch = 1
    else:
        start_epoch = 1

    # Start training
    for epoch in range(start_epoch, num_epoch+1):
        lr = adjust_learning_rate(epoch - 1)
        for param_group in optimizer_D.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer_G.param_groups:
            param_group["lr"] = lr
        print("Epoch={}, lr={}".format(epoch, lr))
        for i, (imgHR, imgLR) in enumerate(data_loader):

            # =================Train generator
            imgHR = Variable(imgHR).cuda()
            imgLR = Variable(imgLR, requires_grad=False).cuda()
            imgLRx = Variable(imgLR).cuda()

            output1 = G1(imgHR)
            output2 = G2(output1)

            # =================Train discriminator
            optimizer_D.zero_grad()
            dis = D(output1.detach())
            lossFake = creterion(dis, Label(dis, False))
            dis = D(imgLR)
            lossReal = creterion(dis, Label(dis, True))
            lossD = lossFake + lossReal
            lossD.backward()
            optimizer_D.step()

            for k in range(1):
                optimizer_G.zero_grad()
                dis = D(output1)
                lossGAN = creterion(dis, Label(dis, True))
                lossCycle = torch.mean(torch.abs(output2 - imgHR))
                lossCycle2 = torch.mean(torch.abs(G1(G2(imgLRx)) - imgLRx))
                lossG = lossGAN + 40 * lossCycle + 10 * lossCycle2
                lossG.backward()
                optimizer_G.step()

            if i % 10 == 9:
                print("===> Epoch[{}]({}/{}): "
                      "LossG: {:.5} (LossGAN:{:.5}, LossCycle:{:.5}, LossCycle2:{:.5}),"
                      " LossD: {:.5}"
                    .format(epoch, i, len(data_loader), lossG, lossGAN, lossCycle, lossCycle2, lossD))
                draw(imgHR, output1, output2, imgLR)
                # draw('input HR %d' % i, (imgHR + 1) * 122.5)
                # draw('G2\'s output', (output2.cpu().detach() + 1) * 122.5)
                # draw('G1\'s output', (output1.cpu().detach() + 1) * 122.5)
                # draw('target', (imgLR + 1) * 122.5)

            # del imgHR, imgLR, output1, output2, lossGAN, lossCycle, lossG,
            # lossFake, lossReal, lossD


        # =================Save checkpoint
        if epoch % 10 == 0:
            model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
            state = {"epoch": epoch, "G1": G1, "D": D, "G2": G2,
                     "optimizerD": optimizer_D, "optimizerG": optimizer_G}
            if not os.path.exists("checkpoint/"):
                os.makedirs("checkpoint/")
            torch.save(state, model_out_path)


if __name__ == "__main__":
    main(ResumePath)
