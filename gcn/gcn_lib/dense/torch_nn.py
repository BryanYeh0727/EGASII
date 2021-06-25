import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


##############################
#    Basic layers
##############################
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # helper selecting normalization layer
    if norm is None:
        return None
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


# class MLP(Seq):
#     def __init__(self, channels, act='relu', norm=None, bias=True):
#         m = []
#         for i in range(1, len(channels)):
#             m.append(Lin(channels[i - 1], channels[i], bias))
#             if act:
#                 m.append(act_layer(act))
#             if norm:
#                 # m.append(norm_layer(norm, channels[-1]))
#                 m.append(norm_layer(norm, channels[i]))
#         super(MLP, self).__init__(*m)


class MLP(nn.Module):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        super(MLP, self).__init__()
        self.conv = 'conv_1x1'
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act:
                m.append(act_layer(act))
            if norm:
                m.append(norm_layer(norm, channels[i]))
        self.body = Seq(*m)

    def forward(self, x, x_0=None, edge_index=None, training=None):
        return self.body(x)


class BasicConv(nn.Module):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        super(BasicConv, self).__init__()
        self.conv = 'conv_1x1'
        # m = []
        # for i in range(1, len(channels)):
        #     m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias))
        #     # m.append(Lin(channels[i - 1], channels[i], bias))
        #     if act:
        #         m.append(act_layer(act))
        #     if norm:
        #         # m.append(norm_layer(norm, channels[-1]))
        #         m.append(norm_layer(norm, channels[i]))
        # self.body = Seq(*m)
        self.nn = Conv2d(channels[0], channels[1], 1, bias=bias)
        if act: self.act = act_layer(act)
        if norm: 
            self.norm = norm_layer(norm, channels[1])
            self.norm_kind = norm

    def forward(self, x, x_0=None, edge_index=None):
        x = self.nn(x)
        if hasattr(self, 'act'): x = self.act(x)
        if hasattr(self, 'norm'):
            if self.norm_kind == 'layer':
                x = x.transpose(1,2).squeeze()
                x = self.norm(x)
                x = x.transpose(1,2).unsqueeze(-1)
            else:
                x = self.norm(x)
        return x

class BasicConv2(nn.Module):
    def __init__(self, channels, act='relu', norm=None, bias=True):
        super(BasicConv2, self).__init__()
        self.conv = 'conv_1x1'
        self.nn = nn.Linear(channels[0], channels[1], bias)
        if act: self.act = act_layer(act)
        if norm: 
            self.norm = norm_layer(norm, channels[1])
            self.norm_kind = norm

    def forward(self, x, x_0=None, edge_index=None):
        # print(x.size())
        x = x.transpose(1,2).squeeze()
        x = self.nn(x)
        if hasattr(self, 'act'): x = self.act(x)

        # x = x.transpose(1,2).unsqueeze(-1)
        # if hasattr(self, 'norm'): x = self.norm(x)
        if hasattr(self, 'norm'):
            if self.norm_kind == 'layer':
                x = self.norm(x)
                x = x.transpose(1,2).unsqueeze(-1)
            else:
                x = x.transpose(1,2).unsqueeze(-1)
                x = self.norm(x)
        else:
            x = x.transpose(1,2).unsqueeze(-1)

        return x


def batched_index_select(inputs, index):
    """

    :param inputs: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """
    batch_size, num_dims, num_vertices, _ = inputs.shape
    k = index.shape[2]
    idx = torch.arange(0, batch_size) * num_vertices
    idx = idx.contiguous().view(batch_size, -1)

    inputs = inputs.transpose(2, 1).contiguous().view(-1, num_dims)
    index = index.contiguous().view(batch_size, -1) + idx.type(index.dtype).to(inputs.device)
    index = index.contiguous().view(-1)

    return torch.index_select(inputs, 0, index).contiguous().view(batch_size, -1, num_dims).transpose(2, 1).contiguous().view(batch_size, num_dims, -1, k)

