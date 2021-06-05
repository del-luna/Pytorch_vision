import torch
import torch.nn as nn
import torch.functional as f


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.expansion = 1   

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcurt = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcurt = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

class ResNet(nn.Module):
    def __init__(self, block, num_blokcs, num_classes=10):
        super(ResNet, self).__init__()
        #After defining only the first layer, the rest are defined by stacking blocks
        #e.g. 
        #self.conv1 = nn.Conv2d(~~~)
        #self.bn1 = nn.BatchNorm2d(~~~)
        #self.layer1 = self._make_layer(block, dim, num_blocks[0], stride=1)
        #self.layer2 = ~~~
        
        #to be define def _make_layer(self, block, planes, num_blocks, stride)
    
