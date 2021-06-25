import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin

class batch_norm(torch.nn.Module):
    def __init__(self, dim_hidden, type_norm, skip_connect=False, num_groups=1,
                 skip_weight=0.005):
        super(batch_norm, self).__init__()
        self.type_norm = type_norm
        self.skip_connect = skip_connect
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden
        if self.type_norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(dim_hidden, momentum=0.3)
        elif self.type_norm == 'group':
            self.bn = torch.nn.BatchNorm1d(dim_hidden*self.num_groups, momentum=0.3)
            self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)
        else:
            pass

    def forward(self, x):
        if self.type_norm == 'None':
            return x
        elif self.type_norm == 'batch':
            # print(self.bn.running_mean.size())
            return self.bn(x)
        elif self.type_norm == 'pair':
            col_mean = x.mean(dim=0)
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = x / rownorm_mean
            return x
        elif self.type_norm == 'group':
            if self.num_groups == 1:
                x_temp = self.bn(x)
            else:
                score_cluster = F.softmax(self.group_func(x), dim=1)
                x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)], dim=1)
                x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)
            x = x + x_temp * self.skip_weight
            return x

        else:
            raise Exception(f'the normalization has not been implemented')


##############################
#    Basic layers
##############################
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act = act_type.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm_type, nc):
    # helper selecting normalization layer
    if norm_type is None:
        return None

    norm = norm_type.lower()
    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc)
    elif norm == 'group':
        layer = nn.GroupNorm(4, nc)
    elif norm == 'dgn':
        layer = batch_norm(nc, 'group', True, 10, 0.005) # 0.01, 0.001
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


class MultiSeq(Seq):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


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
        # print(x.size())
        return self.body(x)


