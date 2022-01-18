import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class ResNet(nn.Module):
    def __init__(self, num_class, model_name="resnet18", pre_train=True):
        super(ResNet, self).__init__()
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pre_train)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pre_train)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pre_train)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc1 = nn.Linear(model.fc.in_features, num_class)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


class BiResNet(nn.Module):
    """
    ResNet + Bilinear Attention
    """

    def __init__(self, num_class, model_name="resnet18", pre_train=True):
        super(BiResNet, self).__init__()
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pre_train)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pre_train)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pre_train)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        self.fc1 = nn.Linear(model.fc.in_features, num_class, bias=False)
        self.fc_bi = nn.Sequential(
            nn.Linear(model.fc.in_features ** 2, model.fc.in_features, bias=False),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(model.fc.in_features, num_class, bias=False)
        self.classifier = nn.Linear(num_class * 2, num_class, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # bilinear pooling-----------------------
        batch_size, C, W, H = x.size()
        xx = x.view(batch_size, C, W ** 2)
        xx = (torch.bmm(xx, torch.transpose(xx, 1, 2)) / W ** 2)
        xx = xx.view(batch_size, -1)
        xx = F.normalize(torch.sign(xx) * torch.sqrt(torch.abs(xx) + 1e-5))
        leb = self.fc_bi(xx)
        # the fc1 output
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)
        # bilinear attention
        x = x + x * leb
        x2 = self.fc2(x)

        out = torch.cat((x1, x2), dim=1)
        out = self.classifier(out)
        return x1, x2, out


class ResNetBP(nn.Module):
    """:param
    Bilinear Pooling based ResNet
    """

    def __init__(self, num_class, model_name="resnet18", pre_train=True):
        super(ResNetBP, self).__init__()
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pre_train)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pre_train)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pre_train)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.fc1 = nn.Linear(model.fc.in_features ** 2, num_class)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        batch_size, C, W, H = x.size()
        x = x.view(batch_size, C, W ** 2)
        x = (torch.bmm(x, torch.transpose(x, 1, 2)) / W ** 2)
        x = x.view(batch_size, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10), p=2, dim=1)

        x = self.fc1(x)
        return x


class DeepGrading(nn.Module):
    """
    ResNet + Bilinear Attention + AuxNet
    """

    def __init__(self, num_class, model_name="resnet18", pre_train=True):
        super(DeepGrading, self).__init__()
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pre_train)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pre_train)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pre_train)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.auxnet = AuxNet(in_channels=2, num_class=num_class)
        self.fc1 = nn.Linear(model.fc.in_features, num_class, bias=False)
        self.fc_bi = nn.Sequential(
            nn.Linear(model.fc.in_features ** 2, model.fc.in_features, bias=False),
            # nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(model.fc.in_features, num_class, bias=False)
        self.classifier = nn.Linear(num_class * 3, num_class, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x, roi):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # bilinear pooling
        batch_size, C, W, H = x.size()
        xx = x.view(batch_size, C, W ** 2)
        xx = (torch.bmm(xx, torch.transpose(xx, 1, 2)) / W ** 2)
        xx = xx.view(batch_size, -1)
        xx = F.normalize(torch.sign(xx) * torch.sqrt(torch.abs(xx) + 1e-5))
        leb = self.fc_bi(xx)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.fc1(x)

        # bilinear attention
        x = x + x * leb
        x2 = self.fc2(x)

        # The AuxNet branch
        roi = self.auxnet(roi)

        out = torch.cat((x1, x2, roi), dim=1)
        out = self.classifier(out)
        return x1, x2, roi, out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.se = SE(channel=out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.se(x)
        return x


class AuxNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(AuxNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
