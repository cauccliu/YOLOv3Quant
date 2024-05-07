from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        
        out_filters = self.backbone.layers_out_filters

        
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Backbone features
        
        x2, x1, x0 = self.backbone(x)

        # Last layer 0
        out0_branch = self.last_layer0[0](x0)
        out0_branch = self.last_layer0[1](out0_branch)
        out0_branch = self.last_layer0[2](out0_branch)
        out0_branch = self.last_layer0[3](out0_branch)
        out0_branch = self.last_layer0[4](out0_branch)
        out0 = self.last_layer0[5](out0_branch)
        out0 = self.last_layer0[6](out0)

        # Last layer 1 - upsample and concatenate
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)

        # Last layer 1
        out1_branch = self.last_layer1[0](x1_in)
        out1_branch = self.last_layer1[1](out1_branch)
        out1_branch = self.last_layer1[2](out1_branch)
        out1_branch = self.last_layer1[3](out1_branch)
        out1_branch = self.last_layer1[4](out1_branch)
        out1 = self.last_layer1[5](out1_branch)
        out1 = self.last_layer1[6](out1)

        # Last layer 2 - upsample and concatenate
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)

        # Last layer 2
        out2 = self.last_layer2[0](x2_in)
        out2 = self.last_layer2[1](out2)
        out2 = self.last_layer2[2](out2)
        out2 = self.last_layer2[3](out2)
        out2 = self.last_layer2[4](out2)
        out2 = self.last_layer2[5](out2)
        out2 = self.last_layer2[6](out2)

        return out0, out1, out2
    
