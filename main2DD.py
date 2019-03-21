import argparse
import itertools
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import ImageDraw, ImageFont
from skimage import measure
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Validation
from ImageFromFolder import *
from model import *
from downSample import *


parser = argparse.ArgumentParser(description="Innovative Super Resolution GAN")
parser.add_argument('weight1', type=int, default=150, nargs='?')
parser.add_argument('weight2', type=int, default=200, nargs='?')
parser.add_argument("-v", "--version", action='version', version='%(prog)s 181231')


def Label(shape, isReal):
    real = torch.Tensor([1]).cuda()
    fake = torch.Tensor([0]).cuda()
    if isReal:
        return real.expand_as(shape)
    else:
        return fake.expand_as(shape)


def draw(inputHR, G1output, G2output, targetLR, image_num):
    inputHR = (inputHR.cpu() + 1) * 122.5
    G1output = (G1output.cpu().detach() + 1) * 122.5
    G2output = (G2output.cpu().detach() + 1) * 122.5
    targetLR = (targetLR.cpu() + 1) * 122.5
    inputHR = inputHR[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    G1output = G1output[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    G2output = G2output[0].numpy().astype(np.uint8).transpose(1, 2, 0)
    targetLR = targetLR[0].numpy().astype(np.uint8).transpose(1, 2, 0)

    ImagePSNR = measure.compare_psnr(G2output, inputHR)
    ImageSSIM = measure.compare_ssim(G2output, inputHR, multichannel=True)

    HR = Image.fromarray(inputHR)
    output1 = Image.fromarray(G1output).resize((256, 256))
    output2 = Image.fromarray(G2output)
    LR = Image.fromarray(targetLR).resize((256, 256))
    images = [HR, output1, output2, LR]
    width = HR.width + output1.width + output2.width + LR.width
    target = Image.new('RGB', (width, HR.height + 50), 'white')
    width = 0

    for image in images:
        target.paste(image, (width, 0))
        width += image.width
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(target)
    draw.text((108, 270), 'input HR', fill=(0, 0, 0), font=font)
    draw.text((364, 270), 'G1 output', fill=(0, 0, 0), font=font)
    draw.text((620, 270), 'G2 output', fill=(0, 0, 0), font=font)
    draw.text((876, 270), 'target LR', fill=(0, 0, 0), font=font)
    draw.text((228, 290), "PSNR: %.3f" % ImagePSNR, fill=(0, 0, 0), font=font)
    draw.text((740, 290), "SSIM: %.3f" % ImageSSIM, fill=(0, 0, 0), font=font)

    if not os.path.exists("training_image_result/"):
        os.makedirs("training_image_result/")
    target.save('training_image_result/2D_1skip_2_{0}_{1}.png'.format(opt.weight1, opt.weight2))


def adjust_learning_rate(epochNo):
    lr = initlr * (0.1 ** (epochNo / steps))
    return lr


class _HighNetD(nn.Module):
    def __init__(self):
        super(_HighNetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 256 x 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 256 x 256
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 256 x 256
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 128 x 128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 128 x 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 64 x 64
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 32 x 32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 32 x 32
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)

        # state size. (512) x 16 x 16
        out = out.view(out.size(0), -1)

        # state size. (512 x 16 x 16)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)


num_epoch = 50
# ResumePath = 'checkpointOY/2D_1skip_2_135.pth'
ResumePath = ''
initlr = 0.0002
start_epoch = 1
steps = 50
vgg_loss = True


def train(resumePath):
    global opt
    opt = parser.parse_args()

    # =================Load data=================
    print("=> loading dataset......")
    img_data = ImageFrom2Folder(
        'Dataset/HRimages',
        'Dataset/LRimages', num=10)
    data_loader = DataLoader(img_data, batch_size=1, shuffle=True)

    # =================Load model=================
    G1 = Net_G()
    G2 = Reverse_G()
    D1 = Net_D()
    D2 = _HighNetD()
    creterion = torch.nn.MSELoss()
    optimizer_G = optim.Adam(itertools.chain(
        G1.parameters(), G2.parameters()), lr=initlr)
    optimizer_D = optim.Adam(itertools.chain(
        D1.parameters(), D2.parameters()), lr=initlr)


    if torch.cuda.is_available():
        D1 = D1.cuda()
        D2 = D2.cuda()
        G1 = G1.cuda()
        G2 = G2.cuda()

    resumePath = 'checkPointOY/2DD_120.pth'
    if resumePath:
        if os.path.isfile(resumePath):
            print("=> loading checkpoint '{}'".format(resumePath))
            checkpoint = torch.load(resumePath)
            start_epoch = checkpoint["epoch"] + 1
            D1.load_state_dict(checkpoint["D1"].state_dict())
            D2.load_state_dict(checkpoint["D2"].state_dict())
            G1.load_state_dict(checkpoint["G1"].state_dict())
            G2.load_state_dict(checkpoint["G2"].state_dict())
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
        for i, (imgHR, imgLR) in enumerate(data_loader, 1):

            imgHR = Variable(imgHR).cuda()
            imgLR = Variable(imgLR, requires_grad=False).cuda()

            output1 = G1(imgHR)
            output2 = G2(output1)

            # =================Train discriminator=================
            optimizer_D.zero_grad()

            dis = D1(output1.detach())
            lossFake = creterion(dis, Label(dis, False))
            dis = D1(imgLR)
            lossReal = creterion(dis, Label(dis, True))
            lossD1 = lossFake + lossReal

            dis = D2(output2.detach())
            lossFake2 = creterion(dis, Label(dis, False))
            dis = D2(imgHR)
            lossReal2 = creterion(dis, Label(dis, True))
            lossD2 = (lossFake2 + lossReal2) * imgHR.shape[-1]

            lossD = lossD1 + lossD2
            lossD.backward()
            optimizer_D.step()

            # =================Train generator=================
            for k in range(1):
                # output1 = G1(imgHR)
                # output2 = G2(output1)

                optimizer_G.zero_grad()

                dis = D1(output1)
                lossGAN = creterion(dis, Label(dis, True))

                dis2 = D2(output2)
                lossGAN2 = creterion(dis2, Label(dis2, True))* imgHR.shape[-1]

                downImage = get_downsimple_Tensor(imgHR.detach().cpu()).cuda()
                loss_idt = creterion(output1, downImage)

                lossCycle = creterion(output2, imgHR)

                lossG = lossGAN + lossGAN2 + loss_idt * opt.weight1 + lossCycle * opt.weight2
                lossG.backward()
                optimizer_G.step()

            # =================Output loss message=================
            if epoch % 5 == 0 and i % len(data_loader) == 0:
                print("===> Epoch[{}]({}/{}):\n LossG: {:.5} (LossGAN:{:.5}, "
                      "LossGAN2:{:.5}, Lossidt:{:.5}, LossCycle:{:.5}),\n LossD1: {:.5}, LossD2: {:.5}\n"
                      .format(epoch, i, len(data_loader),
                              lossG, lossGAN, lossGAN2, loss_idt, lossCycle, lossD1, lossD2))
                # if epoch == 150:
                #     draw(imgHR, output1, output2, imgLR, epoch+i/1000)
                # if vgg_loss:
                #     print('vgg_loss:', content_loss)

        # =================Save checkpoint=================
        if epoch % 5 == 0:
            model_out_path = "checkpointOY/" + "2DD_{0}.pth"\
                .format(epoch)
            state = {"epoch": epoch, "G1": G1, "D1": D1, "D2": D2, "G2": G2}
            torch.save(state, model_out_path)


if __name__ == "__main__":
    # train(ResumePath)

    model = torch.load('checkPointOY/2DD_120.pth')
    Validation.main(model)