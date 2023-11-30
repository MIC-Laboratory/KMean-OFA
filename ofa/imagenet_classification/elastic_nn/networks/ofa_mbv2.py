import random
import torch.nn as nn
import torch.nn.functional as F

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicConvLayer,
    DynamicLinearLayer,
)
from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import (
    DynamicMBConvLayer,
)
from ofa.utils.layers import (
    ConvLayer,
    IdentityLayer,
    LinearLayer,
    MBConvLayer,
    ResidualBlock,
)
from ofa.imagenet_classification.networks import MobileNetV2
from ofa.utils import make_divisible, val2list, MyNetwork

__all__ = ["OFAResNets"]


class OFAMobileNetV2(nn.Module):
    
    def __init__(self, 
    num_classes=100,
    bn_param=(0.1, 1e-5),
    dropout_rate=0,
    depth_list=[1],
    expand_ratio_list=[1],
    width_mult_list=1.0,):
        super(OFAMobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # (expansion, out_planes, num_blocks, stride)
        self.cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
        self.depth_list = depth_list
        self.expand_ratio_list = expand_ratio_list
        self.width_mult_list = width_mult_list
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.ks = [1,16,48,2,34,57,4,36,79,42,2,270,5,2,80,431,185]
        self.num_classes = num_classes
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(DynamicMBConvLayer(
                    [in_planes], 
                    [out_planes], 
                    [3],                
                    self.expand_ratio_list,
                    stride,
                    act_func="relu",
                    expansion=expansion
                    ))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
                # blocks
        

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
        expand_ratio = val2list(e, len(self.layers))

        for block, e in zip(self.layers, expand_ratio):
            if e is not None:
                block.active_expand_ratio = e

    def get_active_subnet(self, preserve_weight=True):
        
        input_channel = 32
        blocks = []
        for stage_id, block_idx in enumerate(range(len(self.layers))):
            blocks.append(
                self.layers[block_idx].get_active_subnet(input_channel, preserve_weight)
            )
            input_channel = self.layers[block_idx].active_out_channel
        classifier = self.linear
        subnet = OFA_MB2_SubNet(blocks,num_classes=self.num_classes)

        return subnet


    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for i,block in enumerate(self.blocks,0):
            if i > len(self.ks) - 1:
                block.re_organize_middle_weights(expand_ratio_stage)
            else:
                block.re_organize_middle_weights(expand_ratio_stage,self.ks[i])


class OFA_MB2_SubNet(nn.Module):
    def __init__(self,blocks,num_classes):
        super(OFA_MB2_SubNet, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # (expansion, out_planes, num_blocks, stride)
        self.cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = nn.Sequential(*blocks)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.ks = [1,16,48,2,34,57,4,36,79,42,2,270,5,2,80,431,185]
    
    def forward(self, x):
                # blocks
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out