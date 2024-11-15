import torch
from torch import nn
import pdb
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, kernel_size=3, padding=1, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=kernel_size, padding=padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, padding=0,normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode="trilinear",align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        upsampling = UpsamplingDeconvBlock

        self.block_five_up = upsampling(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = upsampling(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = upsampling(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = upsampling(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)
        return out_seg, x8_up
 
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        dim_in = 16
        feat_dim = 32
        self.pool = nn.MaxPool3d(3, stride=2)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )
        for class_c in range(2):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(2):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)
        
    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def forward(self, input):
        features = self.encoder(input)
        out_seg, x8_up = self.decoder(features)
        features = self.pool(features[4])
        return out_seg, features


class ECASA_parallel_3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ECASA_parallel_3D, self).__init__()
        self.cSE = eca_layer_3d_v2(num_channels)
        self.sSE = SpatialAttention_3d()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        return output_tensor

class ECASA_parallel_2D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ECASA_parallel_2D, self).__init__()
        self.cSE = eca_layer_2d_v2(num_channels)
        self.sSE = SpatialAttention_2d()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        return output_tensor


class eca_layer_2d_v2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(eca_layer_2d_v2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs(math.log(channel, 2) + 1) / 2)
        k_size = t if t % 2 else (t + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        # Two different branches of ECA module
        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y_avg+y_max)

        return x * y.expand_as(x)

class PPM_v2_allECAv2_3d(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM_v2_allECAv2_3d, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool3d(bin),
                nn.Conv3d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                eca_layer_3d_v2(reduction_dim)
            ))
        self.features = nn.ModuleList(self.features)
        self.end_conv=nn.Conv3d(len(bins)*reduction_dim+in_dim,in_dim,kernel_size=1,bias=False)
        self.end_in=nn.InstanceNorm3d(in_dim)
        self.end_relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='trilinear', align_corners=True))
        end_out=torch.cat(out,1)
        end_out=self.end_conv(end_out)
        end_out=self.end_in(end_out)
        end_out=self.end_relu(end_out)
        return end_out

class PPM_v2_allECAv2_2d(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM_v2_allECAv2_2d, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                eca_layer_2d_v2(reduction_dim)
            ))
        self.features = nn.ModuleList(self.features)
        self.end_conv=nn.Conv2d(len(bins)*reduction_dim+in_dim,in_dim,kernel_size=1,bias=False)
        self.end_in=nn.InstanceNorm2d(in_dim)
        self.end_relu=nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        end_out=torch.cat(out,1)
        end_out=self.end_conv(end_out)
        end_out=self.end_in(end_out)
        end_out=self.end_relu(end_out)
        return end_out


class s4GAN_discriminator_ECSA_2d(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(s4GAN_discriminator_ECSA_2d, self).__init__()

        self.conv1 = nn.Conv2d(num_classes + 1, ndf, kernel_size=3, stride=2, padding=1)  # 160 x 160
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)  # 80 x 80
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1)  # 40 x 40

        self.csSE1 = ECASA_parallel_2D(ndf)
        self.csSE2 = ECASA_parallel_2D(ndf * 2)
        self.csSE3 = ECASA_parallel_2D(ndf * 4)

        self.ppm = PPM_v2_allECAv2_2d(ndf * 4, ndf, [1,2,3,4])

        self.avgpool = nn.AvgPool2d((32,32))
        self.fc = nn.Linear(ndf * 4, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out_maps=[]

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.csSE1(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.csSE2(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.csSE3(x)

        x = self.ppm(x)

        maps = self.avgpool(x)
        out_maps.append(maps)
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))

        return out, out_maps

class s4GAN_discriminator_ECSA_3d(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(s4GAN_discriminator_ECSA_3d, self).__init__()

        self.conv1 = nn.Conv3d(num_classes + 1, ndf, kernel_size=3, stride=2, padding=1)  # 160 x 160
        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)  # 80 x 80
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1)  # 40 x 40

        self.csSE1 = ECASA_parallel_3D(ndf)
        self.csSE2 = ECASA_parallel_3D(ndf * 2)
        self.csSE3 = ECASA_parallel_3D(ndf * 4)

        self.ppm = PPM_v2_allECAv2_3d(ndf * 4, ndf, [1,2,3,4])

        self.avgpool = nn.AvgPool3d((12,12,12))  # Pancreas
        self.fc = nn.Linear(ndf * 4, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout3d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out_maps=[]

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.csSE1(x)
        x = self.drop(x)


        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.csSE2(x)
        x = self.drop(x)


        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.csSE3(x)

        x = self.ppm(x)

        maps = self.avgpool(x)
        out_maps.append(maps)
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))

        return out, out_maps

class s4GAN_discriminator_ECSA_3d_LA(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(s4GAN_discriminator_ECSA_3d_LA, self).__init__()

        self.conv1 = nn.Conv3d(num_classes + 1, ndf, kernel_size=3, stride=2, padding=1)  # 160 x 160
        self.conv2 = nn.Conv3d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)  # 80 x 80
        self.conv3 = nn.Conv3d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1)  # 40 x 40
        self.csSE1 = ECASA_parallel_3D(ndf)
        self.csSE2 = ECASA_parallel_3D(ndf * 2)
        self.csSE3 = ECASA_parallel_3D(ndf * 4)

        self.ppm = PPM_v2_allECAv2_3d(ndf * 4, ndf, [1,2,3,4])


        self.avgpool = nn.AvgPool3d((14,14,10))  # LA
        self.fc = nn.Linear(ndf * 4, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout3d(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out_maps=[]

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.csSE1(x)
        x = self.drop(x)


        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.csSE2(x)
        x = self.drop(x)


        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.csSE3(x)

        x = self.ppm(x)

        maps = self.avgpool(x)
        out_maps.append(maps)
        # conv4_maps = maps
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))

        return out, out_maps


class eca_layer_3d_v2(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(eca_layer_3d_v2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        t = int(abs(math.log(channel,2)+1)/2)
        k_size = t if t%2 else (t+1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        # Two different branches of ECA module
        y_avg = self.conv(y_avg.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y_avg+y_max)

        return x * y.expand_as(x)

class SpatialAttention_3d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_3d, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return input*(x.expand_as(input))

class SpatialAttention_2d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_2d, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return input*(x.expand_as(input))


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=1, normalization='batchnorm', has_dropout=False)
    input = torch.randn(1, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

