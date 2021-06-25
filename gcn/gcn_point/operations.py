import __init__
import torch.nn as nn
from gcn.gcn_lib.dense import GraphConv2d, BasicConv, BasicConv2
# from gcn.gcn_lib.sparse import GraphConv, MLP


search_norm = 'layer' # batch layer


OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity(),
    'conv_1x1': lambda C, stride, affine: BasicConv2([C, C], 'relu', search_norm, bias=False),
    'edge_conv': lambda C, stride, affine: GraphConv2d(C, C, 'edge', 'relu', search_norm, bias=False),
    'mr_conv': lambda C, stride, affine: GraphConv2d(C, C, 'mr', 'relu', search_norm, bias=False),
    'gat': lambda C, stride, affine: GraphConv2d(C, C, 'gat', 'relu', search_norm, bias=False),
    'semi_gcn': lambda C, stride, affine: GraphConv2d(C, C, 'gcn', 'relu', search_norm, bias=False),
    'gin': lambda C, stride, affine: GraphConv2d(C, C, 'gin', 'relu', search_norm, bias=False),
    'sage': lambda C, stride, affine: GraphConv2d(C, C, 'sage', 'relu', search_norm, bias=False),
    'res_sage': lambda C, stride, affine: GraphConv2d(C, C, 'rsage', 'relu', search_norm, bias=False),

    'gcnii': lambda C, stride, layer: GraphConv2d(C, C, 'gcnii', 'relu', None, bias=False, layer=layer), # no norm
    'ginii': lambda C, stride, layer: GraphConv2d(C, C, 'ginii', 'relu', search_norm, bias=False, layer=layer),
    'sageii': lambda C, stride, layer: GraphConv2d(C, C, 'sageii', 'relu', search_norm, bias=True, layer=layer), # bias
    'rsageii': lambda C, stride, layer: GraphConv2d(C, C, 'rsageii', 'relu', search_norm, bias=True, layer=layer), # bias
    'edgeii': lambda C, stride, layer: GraphConv2d(C, C, 'edgeii', 'relu', search_norm, bias=False, layer=layer),
}


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        self.conv = 'skip_connect'

    def forward(self, x, x_0=None, edge_index=None):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.conv = 'none'

    def forward(self, x, x_0=None, edge_index=None):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)




