# deep self-attention network based on VGG
import torch.nn as nn
import torch.nn.functional as F

from net.module.modules import ConcatOutput, weight_init, PPM, encoder, Bottleneck


class exp_4(nn.Module):
    def __init__(self, channel=128):
        super(exp_4, self).__init__()
        self.channel = channel
        self.encoder = encoder()

        self.decoder_0 = self._make_layer(Bottleneck, channel // 4, 3, stride=1)
        self.decoder_1 = self._make_layer(Bottleneck, channel // 4, 3, stride=1)
        self.decoder_2 = self._make_layer(Bottleneck, channel // 4, 3, stride=1)
        self.decoder_3 = self._make_layer(Bottleneck, channel // 4, 3, stride=1)
        self.decoder_4 = self._make_layer(Bottleneck, channel // 4, 3, stride=1)
        self.decoder_5 = self._make_layer(Bottleneck, channel // 4, 3, stride=1)

        self.ppm = PPM(channel)

        self.output_s = ConcatOutput(channel)
        weight_init(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layers = []
        layers.append(block(self.channel, planes, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(self.channel, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]

        x0, x1, x2, x3, x4, x5 = self.encoder(x)

        x_s0 = self.decoder_0(x0)
        x_s1 = self.decoder_1(x1)
        x_s2 = self.decoder_2(x2)
        x_s3 = self.decoder_3(x3)
        x_s4 = self.decoder_4(x4)
        x_s5 = self.decoder_5(x5)

        x_s5 = self.ppm(x_s5)

        pred_s = self.output_s(x_s0, x_s1, x_s2, x_s3, x_s4, x_s5)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)

        return pred_s
