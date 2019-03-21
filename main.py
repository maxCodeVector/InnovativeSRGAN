import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torchvision import models

from ImageFromFolder import *
from model import *
from downSample import *


def Label(shape, isReal):
    real = torch.Tensor([1]).cuda()
    fake = torch.Tensor([0]).cuda()
    if isReal:
        return real.expand_as(shape)
    else:
        return fake.expand_as(shape)

pngi = 0
def draw(inputHR, G1output, G2output, targetLR):
    inputHR = (inputHR.cpu() + 1) * 122.5
    G1output = (G1output.cpu().detach() + 1) * 122.5
    G2output = (G2output.cpu().detach() + 1) * 122.5
    targetLR = (targetLR.cpu() + 1) * 122.5
    inputHR = inputHR[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    G1output = G1output[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    G2output = G2output[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    targetLR = targetLR[0].numpy().astype(np.uint8).transpose(1, 2, 0)

    plt.figure(figsize=(12, 4))
    ax = plt.subplot("141")
    ax.imshow(inputHR)
    ax.set_title("inputHR")

    # plt.figure()
    ax = plt.subplot("142")
    ax.imshow(G1output)
    ax.set_title("G1output")

    # plt.figure()
    ax = plt.subplot("143")
    ax.imshow(G2output)
    ax.set_title("G2output")

    ax = plt.subplot("144")
    ax.imshow(targetLR)
    ax.set_title("targetLR")

    plt.savefig('result/%4d.png'%pngi)
    pngi += 1

def adjust_learning_rate(epochNo):
    lr = initlr * (0.1 ** (epochNo / steps))
    return lr


# def use_vgg():
#     print('===> Loading VGG model')
#     netVGG = models.vgg19()
#     netVGG.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
#     #  model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
#     class _content_model(nn.Module):
#         def __init__(self):
#             super(_content_model, self).__init__()
#             self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
#
#         def forward(self, x):
#             out = self.feature(x)
#             return out
#
#     return _content_model()


num_epoch = 150
# ResumePath = 'checkpoint/model_epoch_96.pth'
ResumePath = ''
initlr = 0.0001
start_epoch = 1
steps = 50
vgg_loss = True


def train(resumePath):

    # =================Load data=================
    print("=> loading dataset......")
    img_data = ImageFrom3Folder(
        'Dataset/HRimages',
        'Dataset/LRimages',
        'Dataset/ReferenceHR')
    data_loader = DataLoader(img_data, batch_size=2, shuffle=True)

    # =================Load model=================
    G1 = Net_G()
    G2 = Reverse_G()
    D = Net_D()
    D2 = Net_D()
    creterion = torch.nn.MSELoss()
    optimizer_G = optim.Adam(itertools.chain(
        G1.parameters(), G2.parameters()), lr=initlr)
    optimizer_D = optim.Adam(itertools.chain(
        D.parameters(), D2.parameters()), lr=initlr)

    # if vgg_loss:
    #     netContent = use_vgg()
    #     netContent = netContent.cuda()

    if torch.cuda.is_available():
        D = D.cuda()
        D2 = D2.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()

    if resumePath:
        if os.path.isfile(resumePath):
            print("=> loading checkpoint '{}'".format(resumePath))
            checkpoint = torch.load(resumePath)
            start_epoch = checkpoint["epoch"] + 1
            D.load_state_dict(checkpoint["D"].state_dict())
            D2.load_state_dict(checkpoint["D2"].state_dict())
            G1.load_state_dict(checkpoint["G1"].state_dict())
            G2.load_state_dict(checkpoint["G2"].state_dict())
            optimizer_G.load_state_dict(checkpoint["optimizerG"].state_dict())
            optimizer_D.load_state_dict(checkpoint["optimizerD"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(resumePath))
            start_epoch = 1
    else:
        start_epoch = 1

    # =================Start training=================
    for epoch in range(start_epoch, num_epoch + 1):
        lr = adjust_learning_rate(epoch)
        for param_group in optimizer_D.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer_G.param_groups:
            param_group["lr"] = lr
        print("Epoch={}, lr={}".format(epoch, lr))
        for i, (imgHR, imgLR, referenceHR) in enumerate(data_loader):

            imgHR = Variable(imgHR).cuda()
            imgLR = Variable(imgLR, requires_grad=False).cuda()
            referenceHR = Variable(imgLR, requires_grad=False).cuda()

            output1 = G1(imgHR)
            output2 = G2(output1)

            # =================Train discriminator=================
            optimizer_D.zero_grad()

            dis = D(output1.detach())
            lossFake = creterion(dis, Label(dis, False))
            dis = D(imgLR)
            lossReal = creterion(dis, Label(dis, True))
            lossD1 = lossFake + lossReal

            dis2 = D2(output2.detach())
            lossFake2 = creterion(dis2, Label(dis2, False))
            dis2 = D2(referenceHR)
            lossReal2 = creterion(dis2, Label(dis2, True))
            lossD2 = lossFake2 + lossReal2

            lossD = lossD1 + lossD2
            lossD.backward()
            optimizer_D.step()

            # =================Train generator=================
            for k in range(1):
                output1 = G1(imgHR)
                output2 = G2(output1)

                optimizer_G.zero_grad()
                dis = D(output1)
                dis2 = D2(output2)
                downImage =Variable(get_downsimple_Tensor(imgHR.detach().cpu()), requires_grad=False).cuda()

                lossGAN = creterion(dis, Label(dis, True))
                lossGAN2 = creterion(dis2, Label(dis2, True))
                lossCycle = creterion(output2, imgHR)
                loss_idt = creterion(output1, downImage)
                # loss_idt2 = creterion(output2, imgHR)
                lossG = lossGAN + loss_idt / 64 + lossGAN2 + 100 * lossCycle
                # if vgg_loss:
                #     content_input = netContent(output2)
                #     content_target = netContent(imgHR).detach()
                #     content_loss = creterion(content_input, content_target) * 1000
                #     netContent.zero_grad()
                #     content_loss.backward(retain_graph=True)
                lossG.backward()
                optimizer_G.step()

            # =================Output loss message=================
            if i % 50 == 0:
                print("===> Epoch[{}]({}/{}): LossG: {:.5} (LossGAN:{:.5}, "
                      "LossGAN2: {:.5}, Lossidt:{:.5}, LossCycle:{:.5}), LossD1: {:.5}, LossD2: {:.5}"
                      .format(epoch, i, len(data_loader),
                              lossG, lossGAN, lossGAN2, loss_idt, lossCycle, lossD1, lossD2))
                # draw(imgHR, output1, output2, imgLR)
                # if vgg_loss:
                #     print('vgg_loss:', content_loss)

        # =================Save checkpoint=================
        if epoch % 10 == 0:
            model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
            state = {"epoch": epoch, "G1": G1, "D": D, "G2": G2, "D2": D2,
                     "optimizerD": optimizer_D, "optimizerG": optimizer_G}
           # if not os.path.exists("checkpoint/"):
            #    os.makedirs("checkpoint/")
            torch.save(state, model_out_path)


if __name__ == "__main__":
    train(ResumePath)
