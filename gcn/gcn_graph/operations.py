import __init__
import torch.nn as nn
from gcn.gcn_lib.sparse import GraphConv, MLP

search_norm = 'layer'

OPS = {
    'none': lambda C, stride, layer: Zero(stride),
    'skip_connect': lambda C, stride, layer: Identity(),
    'conv_1x1': lambda C, stride, layer: MLP([C, C], 'relu', search_norm, bias=False),
    'edge_conv': lambda C, stride, layer: GraphConv(C, C, 'edge', 'relu', search_norm, bias=False),
    'mr_conv': lambda C, stride, layer: GraphConv(C, C, 'mr', 'relu', search_norm, bias=False),
    'gat': lambda C, stride, layer: GraphConv(C, C, 'gat', 'relu', search_norm, bias=False, heads=1, dropout=0.),
    'semi_gcn': lambda C, stride, layer: GraphConv(C, C, 'gcn', 'relu', search_norm, bias=False),
    'gin': lambda C, stride, layer: GraphConv(C, C, 'gin', 'relu', search_norm, bias=False),
    'sage': lambda C, stride, layer: GraphConv(C, C, 'sage', 'relu', search_norm, bias=True),
    'res_sage': lambda C, stride, layer: GraphConv(C, C, 'rsage', 'relu', search_norm, bias=True),
    'sageii': lambda C, stride, layer: GraphConv(C, C, 'sageii', 'relu', search_norm, bias=True, layer=layer),
    'res_sageii': lambda C, stride, layer: GraphConv(C, C, 'rsageii', 'relu', search_norm, bias=True, layer=layer),
    'gcnii': lambda C, stride, layer: GraphConv(C, C, 'gcnii', 'relu', search_norm, bias=False, layer=layer),
    'ginii': lambda C, stride, layer: GraphConv(C, C, 'ginii', 'relu', search_norm, bias=False, layer=layer),
    'gatii': lambda C, stride, layer: GraphConv(C, C, 'gatii', 'relu', search_norm, bias=False, layer=layer, heads=1, dropout=0.),
    'gatiia': lambda C, stride, layer: GraphConv(C, C, 'gatiia', 'relu', search_norm, bias=False, layer=layer, heads=1, dropout=0.),
    'appnp': lambda C, stride, layer: GraphConv(C, C, 'appnp', 'relu', search_norm, bias=False, dropout=0.),
    'clustergcn': lambda C, stride, layer: GraphConv(C, C, 'clustergcn', 'relu', search_norm, bias=False),
    'clustergcnii': lambda C, stride, layer: GraphConv(C, C, 'clustergcnii', 'relu', search_norm, bias=False, layer=layer),
}


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()
        self.conv = 'skip_connect'

    def forward(self, x, x_0=None, edge_index=None, training=None):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.conv = 'none'

    def forward(self, x, x_0=None, edge_index=None, training=None):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
