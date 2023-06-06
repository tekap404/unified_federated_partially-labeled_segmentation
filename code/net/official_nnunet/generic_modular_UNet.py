import torch
from nnunet.network_architecture.custom_modules.conv_blocks import StackedConvLayers
from .generic_UNet import Upsample
from nnunet.network_architecture.neural_network import SegmentationNetwork
from torch import nn
import numpy as np
from torch.cuda.amp import autocast as autocast
from copy import deepcopy
from nnunet.network_architecture.custom_modules.helperModules import Identity
"""
The idea behind this modular U-net ist that we decouple encoder and decoder and thus make things a) a lot more easy to 
combine and b) enable easy swapping between segmentation or classification mode of the same architecture
"""


def get_default_network_config(dim=2, dropout_p=None, nonlin="LeakyReLU", norm_type="bn"):
    """
    returns a dictionary that contains pointers to conv, nonlin and norm ops and the default kwargs I like to use
    :return:
    """
    props = {}
    if dim == 2:
        props['conv_op'] = nn.Conv2d
        props['dropout_op'] = nn.Dropout2d
    elif dim == 3:
        props['conv_op'] = nn.Conv3d
        props['dropout_op'] = nn.Dropout3d
    else:
        raise NotImplementedError

    if norm_type == "bn":
        if dim == 2:
            props['norm_op'] = nn.BatchNorm2d
        elif dim == 3:
            props['norm_op'] = nn.BatchNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True, 'track_running_stats': False}
    elif norm_type == "in":
        if dim == 2:
            props['norm_op'] = nn.InstanceNorm2d
        elif dim == 3:
            props['norm_op'] = nn.InstanceNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True, 'track_running_stats': False}
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_op'] = None
        props['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_op_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_op_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': True}  # kernel size will be set by network!

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    elif nonlin == "GELU":
        props['nonlin'] = nn.GELU
        props['nonlin_kwargs'] = {}
    elif nonlin == "PReLU":
        props['nonlin'] = nn.PReLU
        props['nonlin_kwargs'] = {}
    else:
        raise ValueError

    return props

# for PreActResidualUNet
class PlainConvUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, default_return_skips=True,
                 max_num_features=480, deep_supervision=False, upscale_logits=False, num_classes=4):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)
        this one includes the bottleneck layer!
        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(PlainConvUNetEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props
        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this
        current_input_features = input_channels

        for stage in range(num_stages):
            current_output_features = min(int(base_num_features * feat_map_mul_on_downscale ** stage), max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = StackedConvLayers(current_input_features, current_output_features, current_kernel_size,
                                              props, num_blocks_per_stage[stage], current_pool_kernel_size)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_output_features

    def forward(self, x, return_skips=None):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        for i, s in enumerate(self.stages):
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

class Deep_supervision(nn.Module):
    def __init__(self, conv_kernel_sizes, base_num_features, feat_map_mul_on_downscale, pool_op_kernel_sizes, 
                    max_num_features, num_classes, props, upscale_logits=False, cfg=None):
        super(Deep_supervision, self).__init__()
        self.num_stages = len(conv_kernel_sizes)
        self.props = props
        self.cfg = cfg
        
        self.deep_supervision_outputs = []
        cum_upsample = 1
        for stage in range(self.num_stages):
            current_output_features = min(int(base_num_features * feat_map_mul_on_downscale ** stage), max_num_features)
            current_pool_kernel_size = pool_op_kernel_sizes[stage]
            # only used for upsample_logits
            cum_upsample *= current_pool_kernel_size[0]
            upsample_mode = "trilinear"
            if stage != 0 and stage != self.num_stages-1:
                seg_layer = self.props['conv_op'](current_output_features, num_classes, 1, 1, 0, 1, 1, False)
                if upscale_logits:
                    upsample = Upsample(scale_factor=cum_upsample, mode=upsample_mode)
                    self.deep_supervision_outputs.append(nn.Sequential(seg_layer, upsample))
                else:
                    self.deep_supervision_outputs.append(seg_layer)

        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, x):
        seg_outputs = []

        for i, s in enumerate(x):
            if i != 0 and i != len(x)-1:
                tmp = self.deep_supervision_outputs[i-1](s)
                seg_outputs.append(tmp)

        seg_outputs = torch.stack(seg_outputs)

        return seg_outputs

class PlainConvUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, network_props=None, deep_supervision=False,
                 upscale_logits=False, cfg=None):
        super(PlainConvUNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.cfg = cfg
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if network_props is None:
            self.props = previous.props
        else:
            self.props = network_props

        if self.props['conv_op'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
        elif self.props['conv_op'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # we have one less as the first stage here is what comes after the
        # bottleneck
        self.tus = []
        self.stages = []
        self.deep_supervision_outputs = []

        # only used for upsample_logits
        cum_upsample = np.cumprod(np.vstack(self.stage_pool_kernel_size), axis=0).astype(int)

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]


            self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                    previous_stage_pool_kernel_size[s + 1], bias=False))
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(StackedConvLayers(2 * features_skip, features_skip,
                                                previous_stage_conv_op_kernel_size[s], self.props,
                                                num_blocks_per_stage[i]))
            
        self.segmentation_output = self.props['conv_op'](features_skip, num_classes, 1, 1, 0, 1, 1, False)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)
        self.deep_supervision_outputs = nn.ModuleList(self.deep_supervision_outputs)

    def forward(self, skips, deep_outputs=None):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]
        seg_outputs = []

        x = skips[0]  # this is the bottleneck

        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)

        segmentation = self.segmentation_output(x)
        if self.training == True and self.deep_supervision:
            seg_outputs = torch.cat((deep_outputs, segmentation.unsqueeze(0)), 0).permute(1,0,2,3,4,5)
            return seg_outputs
        else:
            return segmentation


class PlainConvUNet(SegmentationNetwork):

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, client_num=4, cfg=None, initializer=None):
        super(PlainConvUNet, self).__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes
        self.cfg = cfg
        self.client_num = client_num
        self.conv_kernel_sizes = conv_kernel_sizes

        self.encoder = PlainConvUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                                feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                                props, default_return_skips=True, max_num_features=max_features,
                                                deep_supervision=deep_supervision, upscale_logits=upscale_logits, num_classes=num_classes)
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                        deep_supervision, upscale_logits, self.cfg)
        self.supervision = Deep_supervision(conv_kernel_sizes, base_num_features, feat_map_mul_on_downscale, pool_op_kernel_sizes, 
                                                max_features, num_classes, props, upscale_logits, cfg)
        
        if initializer is not None:
            self.apply(initializer)

        self.deep_supervision = deep_supervision

    def forward(self, x):
        with autocast():
            
            skips = self.encoder(x)

            if self.training == True and self.deep_supervision == True:
                seg_outputs = self.supervision(skips)
                out = self.decoder(skips, seg_outputs)
            else:
                out = self.decoder(skips)
        return out