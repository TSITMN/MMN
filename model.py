import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def my_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.333)
        nn.init.constant_(m.bias, 0.0)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # Reshape input from (H, W, C) to (C, H, W)
        x = x.permute(2, 0, 1).unsqueeze(0) if len(x.shape) == 3 else x
        # 通道注意力机制
        max_out = self.max_pool(x)
        max_out = self.mlp(max_out.view(max_out.size(0), -1))
        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out.view(avg_out.size(0), -1))
        channel_out = self.sigmoid(max_out + avg_out)
        channel_out = channel_out.view(x.size(0), x.size(1), 1, 1)
        channel_out = channel_out * x

        # 空间注意力机制
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        mean_out = torch.mean(channel_out, dim=1, keepdim=True)
        spatial_out = torch.cat((max_out, mean_out), dim=1)
        spatial_out = self.sigmoid(self.conv(spatial_out))
        out = spatial_out * channel_out
        # Reshape output back to (H, W, C)
        out = out.squeeze(0).permute(1, 2, 0) if len(x.shape) == 3 else out
        return out

class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.fc = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.bn(F.relu(self.fc(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, module_type, arch='resnet50'):
        super(FeatureExtractor, self).__init__()
        if module_type == 'visible':
            model = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
            # 使用可见光模型的第一层
            self.module = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool
            )
        elif module_type == 'thermal':
            model = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
            # 使用热成像模型的第一层
            self.module = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool
            )

    def forward(self, x):
        x = self.module(x)
        return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.visible_module = FeatureExtractor('visible', arch=arch)
        self.thermal_module = FeatureExtractor('thermal', arch=arch)
        # self.cbam = CBAM(in_channel=3)
        self.base_resnet = base_resnet(arch=arch)

        self.encoder1 = Encoder(3, 1)
        self.encoder2 = Encoder(3, 1)
        self.decoder = Decoder(1, 3)  
        
        self.l2norm = Normalize(2)     
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        pool_dim = 2048
        self.bottlenecks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for i in range(4):  
            bottleneck = nn.BatchNorm1d(pool_dim)
            bottleneck.bias.requires_grad_(False)  # no shift
            bottleneck.apply(weights_init_kaiming)
            self.bottlenecks.append(bottleneck)
            
            classifier = nn.Linear(pool_dim, class_num, bias=False)
            classifier.apply(weights_init_classifier)
            self.classifiers.append(classifier)
            
    def forward(self, x1, x2):

        gray1 = self.encoder1(x1)
        gray2 = self.encoder2(x2)
        gray = torch.cat((gray1, gray2), dim=0)
        gray = self.decoder(gray)

        gray1, gray2 = torch.chunk(gray, 2, 0)
        
        # # Processing with visible and thermal modules
        x1 = self.visible_module(torch.cat((x1, gray1), dim=0))
        x2 = self.thermal_module(torch.cat((x2, gray2), dim=0))

        # # Apply CBAM
        # # x1 = self.cbam(x1)
        # # x2 = self.cbam(x2)

        xo = torch.cat((x1, x2), 0)
        # Concatenate the outputs
        x = torch.cat((x1, x2), dim=0)

        # shared block
        x = self.base_resnet(x)

        x_parts = torch.chunk(x, 4, 2)
    
        x_parts = [self.avgpool(x_part) for x_part in x_parts]
        x_parts = [x_part.view(x_part.size(0), x_part.size(1)) for x_part in x_parts] 
        feats = [self.bottlenecks[i](x_parts[i]) for i in range(4)]
        outputs = [self.classifiers[i](feats[i]) for i in range(4)]

        if self.training:
            return (*x_parts, *outputs, [xo, gray])
        else:
            all_x = torch.cat(x_parts, 1)
            all_feats = torch.cat(feats, 1)
            return self.l2norm(all_x), self.l2norm(all_feats)
            
            
            
            
            
            
            
            
            
            
            