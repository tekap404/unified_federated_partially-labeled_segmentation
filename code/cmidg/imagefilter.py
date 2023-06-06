# GIN
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast as autocast

# conv + Leaky-ReLU
class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, cfg, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, **kwcfg):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        assert requires_grad == False
        self.cfg = cfg

    def forward(self, x_in, device, requires_grad = False):
        """
        cfg:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # if self.cfg.use_multi_gpu == True:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.multi_gpu
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "'" + str(self.cfg.gpu) + "'"
        with autocast():
            # random size of kernel
            idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
            k = self.scale_pool[idx_k[0]]

            nb, nc, nd, nx, ny = x_in.shape

            ker = torch.randn([self.out_channel * nb, self.in_channel , k, k, k], requires_grad = self.requires_grad).half().cuda(device)#.cuda()
            shift = (torch.randn([self.out_channel * nb, 1, 1, 1], requires_grad = self.requires_grad) * 1.0).half().cuda(device)#.cuda()

            x_in = x_in.view(1, nb * nc, nd, nx, ny)
            x_conv = F.conv3d(x_in, ker, stride = 1, padding = k // 2, dilation = 1, groups = nb)
            x_conv = x_conv + shift
            if self.use_act:
                x_conv = F.leaky_relu(x_conv)

            x_conv = x_conv.view(nb, self.out_channel, nd, nx, ny)
        return x_conv

# GIN网络————n_layer层的conv + Leaky-ReLU，alphas加权，re-norm
class GINGroupConv(nn.Module):
    def __init__(self, cfg, out_channel = 1, in_channel = 1, interm_channel = 1, scale_pool = [1, 3], n_layer = 4, out_norm = 'frob', **kwcfg):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.cfg = cfg

        # if self.cfg.use_multi_gpu == True:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.multi_gpu
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "'" + str(self.cfg.gpu) + "'"
            
        self.layers.append(
            GradlessGCReplayNonlinBlock(cfg, out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, layer_id = 0).cuda()
                )
        for ii in range(n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock(cfg, out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = ii + 1).cuda()
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock(cfg, out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = n_layer - 1, use_act = False).cuda()
                )

        self.layers = nn.ModuleList(self.layers)


    def forward(self, x_in, device):
        # if self.cfg.use_multi_gpu == True:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.multi_gpu
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "'" + str(self.cfg.gpu) + "'"

        with autocast():
            nb, nc, nd, nx, ny = x_in.shape

            alphas = torch.rand(nb)[:, None, None, None, None] # nb, 1, 1, 1
            alphas = alphas.repeat(1, nc, 1, 1, 1).half().cuda(device)#.cuda() # nb, nc, 1, 1

            x = self.layers[0](x_in, device)
            for blk in self.layers[1:]:
                x = blk(x, device)
            mixed = alphas * x + (1.0 - alphas) * x_in

            if self.out_norm == 'frob':
                _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
                _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
                _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
                _self_frob = _self_frob[:, None, None, None, None].repeat(1, self.out_channel, 1, 1, 1)
                mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed


