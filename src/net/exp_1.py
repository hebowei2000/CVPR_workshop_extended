# deep self-attention network based on VGG
import torch.nn as nn
import torch.nn.functional as F

from net.module.modules import Reduction, ConcatOutput, weight_init, PPM, encoder


class exp_1(nn.Module):
    def __init__(self, channel=128):
        super(exp_1, self).__init__()
        self.encoder = encoder()

        self.reduce_s0 = Reduction(128, channel)
        self.reduce_s1 = Reduction(256, channel)
        self.reduce_s2 = Reduction(512, channel)
        self.reduce_s3 = Reduction(1024, channel)
        self.reduce_s4 = Reduction(2048, channel)

        self.ppm = PPM(channel)

        self.output_s = ConcatOutput(channel)
        weight_init(self)

    def forward(self, x):
        size = x.size()[2:]

        x0, x1, x2, x3, x4 = self.encoder(x)

        x_s0 = self.reduce_s0(x0)
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_s4 = self.ppm(x_s4)

        pred_s = self.output_s(x_s0, x_s1, x_s2, x_s3, x_s4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)

        return pred_s
