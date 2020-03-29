# deep self-attention network based on VGG
import torch.nn as nn
import torch.nn.functional as F

from net.module.modules import Reduction, ConcatOutput, weight_init, encoder_w,encoder_w_dropout,encoder_w_coord_dropout, Bottle2neck, ASPP, ConcatFusion


class exp_17(nn.Module):
    def __init__(self, channel=128):
        super(exp_17, self).__init__()
        self.channel = channel
        self.encoder = encoder_w_dropout()

        self.reduce_s0 = Reduction(64, channel)
        self.reduce_s1 = Reduction(128, channel)
        self.reduce_s2 = Reduction(256, channel)
        self.reduce_s3 = Reduction(512, channel)
        self.reduce_s4 = Reduction(1024, channel)
        self.reduce_s5 = Reduction(2048, channel)

        self.decoder_0 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_1 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_2 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_3 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_4 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_5 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)

        self.decoder_6 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_7 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_8 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_9 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_10 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_11 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)

        self.decoder_12 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_13 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_14 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_15 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_16 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)
        self.decoder_17 = self._make_layer(Bottle2neck, channel // 4, 3, stride=1)

        self.aspp_1 = ASPP(in_channel=2048, depth=2048)
        self.aspp_2 = ASPP(in_channel=channel, depth=channel)
        self.aspp_3 = ASPP(in_channel=channel, depth=channel)

        self.stage_1 = ConcatFusion(channel)
        self.stage_2 = ConcatFusion(channel)
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

        x5 = self.aspp_1(x5)

        x_s0 = self.decoder_0(self.reduce_s0(x0))
        x_s1 = self.decoder_1(self.reduce_s1(x1))
        x_s2 = self.decoder_2(self.reduce_s2(x2))
        x_s3 = self.decoder_3(self.reduce_s3(x3))
        x_s4 = self.decoder_4(self.reduce_s4(x4))
        x_s5 = self.decoder_5(self.reduce_s5(x5))

        x_s0, x_s1, x_s2, x_s3, x_s4, x_s5 = self.stage_1(x_s0, x_s1, x_s2, x_s3, x_s4, x_s5)

        x_s5 = self.aspp_2(x_s5)

        x_s0 = self.decoder_6(x_s0)
        x_s1 = self.decoder_7(x_s1)
        x_s2 = self.decoder_8(x_s2)
        x_s3 = self.decoder_9(x_s3)
        x_s4 = self.decoder_10(x_s4)
        x_s5 = self.decoder_11(x_s5)

        x_s0, x_s1, x_s2, x_s3, x_s4, x_s5 = self.stage_2(x_s0, x_s1, x_s2, x_s3, x_s4, x_s5)

        x_s5 = self.aspp_3(x_s5)

        x_s0 = self.decoder_12(x_s0)
        x_s1 = self.decoder_13(x_s1)
        x_s2 = self.decoder_14(x_s2)
        x_s3 = self.decoder_15(x_s3)
        x_s4 = self.decoder_16(x_s4)
        x_s5 = self.decoder_17(x_s5)

        pred_s = self.output_s(x_s0, x_s1, x_s2, x_s3, x_s4, x_s5)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)

        return pred_s
