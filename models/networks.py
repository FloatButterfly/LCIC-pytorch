import functools

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models, transforms
from collections import namedtuple
import numpy as np
import os
# from pytorch_msssim import SSIM, ssim


###############################################################################
# Functions
###############################################################################
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_G(opt, gpu_ids=[]):
    input_nc = opt.input_nc
    output_nc = opt.output_nc
    nz = opt.nz
    ngf = opt.ngf
    netG = opt.netG
    norm = opt.norm
    nl = opt.nl
    use_dropout = opt.use_dropout
    init_type = opt.init_type
    where_add = opt.where_add
    upsample = opt.upsample
    use_res = opt.use_res

    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if netG == 'unet_256' and where_add == 'all':
        net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                             use_dropout=use_dropout, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


def define_D(input_nc, ndf, netD,
             norm='batch', nl='lrelu',
             use_sigmoid=False, init_type='xavier', num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer,
                             use_sigmoid=use_sigmoid, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, gpu_ids)


def define_E(input_nc, ndf, netE, output_nc=1, norm='batch', label_nc=0,
             init_type='xavier', gpu_ids=[], vaeLike=False):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)

    if netE == 'resnet_256':
        net = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                       nl_layer=nl_layer, vaeLike=vaeLike)

    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf_i = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf_i, n_layers, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                            stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=kw, stride=2, padding=padw)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                                             kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    # model[0] output 30*30*1 model[1] output 6*6*1
    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        mean = 0.0
        for i in range(self.num_D):
            output = self.model[i](down)
            mean += output.mean()
            result.append(output)
            if i != self.num_D - 1:
                down = self.down(down)  # 256*256*3 -> 128*128*3
        mean = mean / self.num_D * 1.0
        return result, mean


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor
        self.gan_mode = gan_mode
        # self.register_buffer('real_label', torch.tensor(target_real_label))
        # self.register_buffer('fake_label', torch.tensor(target_fake_label))
        # self.loss = nn.MSELoss() if mse_loss else nn.BCELoss

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input).cuda()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real).cuda()
            return F.mse_loss(input, target_tensor)
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck

def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'add_output_padding':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1, output_padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [spectral_norm(nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                                         stride=1, padding=padw, bias=True))]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [spectral_norm(nn.Conv2d(inplanes, outplanes,
                                         kernel_size=1, stride=1, padding=0, bias=True))]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [spectral_norm(conv3x3(inplanes, outplanes))]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsampleConv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [spectral_norm(conv3x3(outplanes, outplanes))]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsampleConv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


# set norm_layer is none
class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [spectral_norm(conv3x3(inplanes, inplanes))]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        if n_blocks == 6:
            max_ndf = n_blocks
        else:
            max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz

        # construct unet structure
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, num_downs=num_downs)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                      upsample=upsample)
        for i in range(num_downs - 6):  # 8-6=2 0,1
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                          upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero', num_downs=0):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            if num_downs == 4:
                upsample = "add_output_padding"
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)
