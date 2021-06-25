import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from .torch_nn import MLP, act_layer, norm_layer
from .torch_edge import DilatedKnnGraph
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from typing import Union, Tuple, Callable, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, NoneType

from torch import Tensor
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, matmul, set_diag, sum
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform, reset, zeros
from math import log

from torch_geometric.typing import Adj, Size, OptTensor


class GCN2Conv(MessagePassing):
    r"""The graph convolutional operator with initial residual connections and
    identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
        \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
        \mathbf{\Theta} \right)

    with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2}`, where
    :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
    matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
    and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
    Here, :math:`\alpha` models the strength of the initial residual
    connection, while :math:`\beta` models the strength of the identity
    mapping.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        channels (int): Size of each input and output sample.
        alpha (float): The strength of the initial residual connection
            :math:`\alpha`.
        theta (float, optional): The hyperparameter :math:`\theta` to compute
            the strength of the identity mapping
            :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
            (default: :obj:`None`)
        layer (int, optional): The layer :math:`\ell` in which this module is
            executed. (default: :obj:`None`)
        shared_weights (bool, optional): If set to :obj:`False`, will use
            different weight matrices for the smoothed representation and the
            initial residual ("GCNII*"). (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCN2Conv, self).__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        # x.mul_(1 - self.alpha)
        # x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=(1 - self.alpha)*(1. - self.beta),
                              alpha=self.beta)
            out += torch.addmm(x_0, x_0, self.weight2, beta=(self.alpha)*(1. - self.beta),
                               alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, alpha={}, beta={})'.format(self.__class__.__name__,
                                                  self.channels, self.alpha,
                                                  self.beta)


class EdgConv(tg.nn.EdgeConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)

    def forward(self, x, x_0, adj_t):
        if isinstance(adj_t, SparseTensor):
            row, col, edge_attr = adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        elif isinstance(adj_t, Tensor):
            edge_index = adj_t
        return super(EdgConv, self).forward(x, edge_index)


class GATConvIIa(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 alpha: float = None, theta: float = None, layer: int = None, norm = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConvIIa, self).__init__(node_dim=0, **kwargs)

        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.weight1 = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.weight2 = Parameter(torch.Tensor(in_channels, heads * out_channels))

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.norm = norm_layer(norm, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], x_0, edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Tensor, bool, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, SparseTensor, bool, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Tensor, bool, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, SparseTensor, bool, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = x.view(-1, H, C)
            # x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)


        out = torch.addmm(out, out, self.weight1, beta=(1 - self.alpha)*(1. - self.beta),
                              alpha=self.beta)
        out += torch.addmm(x_0, x_0, self.weight2, beta=(self.alpha)*(1. - self.beta),
                        alpha=self.beta)


        if self.bias is not None:
            out += self.bias


        if self.norm is not None:
            out = self.norm(out.relu())
        else:
            out = out.relu()

        return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GATConvII(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 alpha: float = None, theta: float = None, layer: int = None, norm = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConvII, self).__init__(node_dim=0, **kwargs)

        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.weight1 = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.weight2 = Parameter(torch.Tensor(in_channels, heads * out_channels))

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.norm = norm_layer(norm, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], x_0, edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Tensor, bool, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, SparseTensor, bool, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Tensor, bool, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, SparseTensor, bool, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'

            x_ = torch.addmm(x, x, self.weight1, beta=(1 - self.alpha)*(1. - self.beta),
                              alpha=self.beta)
            x_ += torch.addmm(x_0, x_0, self.weight2, beta=(self.alpha)*(1. - self.beta),
                            alpha=self.beta)

            x_l = x_r = x_.view(-1, H, C)
            # x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.norm is not None:
            out = self.norm(out.relu())
        else:
            out = out.relu()

        return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GATConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels,  act='relu', norm=None, bias=True, heads=8, dropout=0.):
        super(GATConv, self).__init__()
        self.gconv = tg.nn.GATConv(in_channels, out_channels, heads, bias=bias, dropout=dropout)
        m =[]
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, x_0, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out

class SAGEConvII(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, relative = False, 
                 alpha: float = None, theta: float = None, layer: int = None, norm = None, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConvII, self).__init__(**kwargs)

        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.relative = relative

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.weight1 = Parameter(torch.Tensor(in_channels[0], in_channels[0]))
        self.weight2 = Parameter(torch.Tensor(in_channels[0], in_channels[0]))
        
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.norm = norm_layer(norm, in_channels[0])

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self.lin_r.reset_parameters()
        uniform(self.out_channels, self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], x_0, adj_t: Adj,
                size: Size = None) -> Tensor:
        """"""
        if self.relative and isinstance(adj_t, SparseTensor):
            row, col, edge_attr = adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = adj_t

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        x_l = self.propagate(edge_index, x=x, size=size)
        # out = self.lin_l(out)

        out = torch.addmm(x_l, x_l, self.weight1, beta=(1 - self.alpha)*(1. - self.beta),
                              alpha=self.beta)
        out += torch.addmm(x_0, x_0, self.weight2, beta=(self.alpha)*(1. - self.beta),
                            alpha=self.beta)


        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)


        if self.norm is not None:
            out = self.norm(out.relu())
        else:
            out = out.relu()

        if self.bias is not None:
            out = out + self.bias


        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i, x_j: Tensor) -> Tensor:
        # if self.relative: return x_j - x_i
        # else: return x_j
        return (x_j - x_i) if self.relative else x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, relative = False, norm = None, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.relative = relative

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=False)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.norm = norm_layer(norm, in_channels[0])

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        uniform(self.out_channels, self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], x_0, adj_t: Adj,
                size: Size = None) -> Tensor:
        """"""
        if self.relative and isinstance(adj_t, SparseTensor):
            row, col, edge_attr = adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = adj_t

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        if self.norm is not None:
            out = self.norm(out.relu())
        else:
            out = out.relu()

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_i, x_j: Tensor) -> Tensor:
        # if self.relative: return x_j - x_i
        # else: return x_j
        return (x_j - x_i) if self.relative else x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SemiGCNConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(SemiGCNConv, self).__init__()
        self.gconv = tg.nn.GCNConv(in_channels, out_channels, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, x_0, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class GCNIIConv(nn.Module):
    """
    GCNII convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, layer=1, alpha=0.5, theta=1.0):
        super(GCNIIConv, self).__init__()
        self.gconv = GCN2Conv(channels=in_channels, alpha=alpha, theta=theta, layer=layer, shared_weights=False)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, x_0, edge_index):
        out = self.gconv(x, x_0, edge_index)
        out = self.unlinear(out)
        return out

class GINConvII(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, channels, eps: float = 0., train_eps: bool = False,
                 alpha: float = None, theta: float = None, layer: int = None, 
                 norm = None, bias = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConvII, self).__init__(**kwargs)
        self.channels = channels

        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter('bias', None)

        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)

        self.weight1 = Parameter(torch.Tensor(channels, channels))
        self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.norm = norm_layer(norm, channels)

        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self.eps.data.fill_(self.initial_eps)
        uniform(self.channels, self.bias)


    def forward(self, x: Union[Tensor, OptPairTensor], x_0, edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        x_l = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            x_l += (1 + self.eps) * x_r

        out = torch.addmm(x_l, x_l, self.weight1, beta=(1 - self.alpha)*(1. - self.beta),
                              alpha=self.beta)
        out += torch.addmm(x_0, x_0, self.weight2, beta=(self.alpha)*(1. - self.beta),
                            alpha=self.beta)

        if self.bias is not None:
            out = out + self.bias

        if self.norm is not None:
            out = self.norm(out.relu())
        else:
            out = out.relu()

        return out


    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class GinConv(tg.nn.GINConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
        super(GinConv, self).__init__(MLP([in_channels, out_channels], act, norm, bias))

    def forward(self, x, x_0, edge_index):
        return super(GinConv, self).forward(x, edge_index)


class APPNPConv(nn.Module):
    def __init__(self, K=10, alpha=0.1, dropout=0.):
        super(APPNPConv, self).__init__()
        self.gconv = tg.nn.APPNP(K=K, alpha=alpha, dropout=dropout)

    def forward(self, x, x_0, edge_index):
        out = self.gconv(x, edge_index)
        return out


class ClusterGCNConvII(MessagePassing):
    r"""The ClusterGCN graph convolutional operator from the
    `"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
    Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \left( \mathbf{\hat{A}} + \lambda \cdot
        \textrm{diag}(\mathbf{\hat{A}}) \right) \mathbf{X} \mathbf{W}_1 +
        \mathbf{X} \mathbf{W}_2

    where :math:`\mathbf{\hat{A}} = {(\mathbf{D} + \mathbf{I})}^{-1}(\mathbf{A}
    + \mathbf{I})`.

    Args:
        in_channels (int or tuple): Size of each input sample.
        out_channels (int): Size of each output sample.
        diag_lambda: (float, optional): Diagonal enhancement value
            :math:`\lambda`. (default: :obj:`0.`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 diag_lambda: float = 0., add_self_loops: bool = True,
                 bias: bool = True, 
                 alpha: float = None, theta: float = None, layer: int = None, 
                 norm = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ClusterGCNConvII, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diag_lambda = diag_lambda
        self.add_self_loops = add_self_loops

        self.root_weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))

        self.norm = norm_layer(norm, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        glorot(self.root_weight)
        zeros(self.bias)


    def forward(self, x: Tensor, x_0, edge_index: Adj, size: Size = None) -> Tensor:
        """"""
        edge_weight: OptTensor = None
        if isinstance(edge_index, Tensor):
            num_nodes = size[1] if size is not None else x.size(self.node_dim)
            if self.add_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

            row, col = edge_index[0], edge_index[1]
            deg_inv = 1. / degree(col, num_nodes=num_nodes).clamp_(1.)

            edge_weight = deg_inv[col]
            edge_weight[row == col] += self.diag_lambda * deg_inv

        elif isinstance(edge_index, SparseTensor):
            if self.add_self_loops:
                edge_index = set_diag(edge_index)

            col, row, _ = edge_index.coo()  # Transposed.
            deg_inv = 1. / sum(edge_index, dim=1).clamp_(1.)

            edge_weight = deg_inv[col]
            edge_weight[row == col] += self.diag_lambda * deg_inv
            edge_index = edge_index.set_value(edge_weight, layout='coo')

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x_l = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        # out = out @ self.weight + x @ self.root_weight

        out = torch.addmm(x_l, x_l, self.weight1, beta=(1 - self.alpha)*(1. - self.beta),
                              alpha=self.beta)
        out += torch.addmm(x_0, x_0, self.weight2, beta=(self.alpha)*(1. - self.beta),
                            alpha=self.beta)

        if self.bias is not None:
            out += self.bias

        if self.norm is not None:
            out = self.norm(out.relu())
        else:
            out = out.relu()

        return out


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, diag_lambda={})'.format(self.__class__.__name__,
                                                   self.in_channels,
                                                   self.out_channels,
                                                   self.diag_lambda)


class ClusterGcnConv(nn.Module):
    def __init__(self, in_channels, out_channels, diag_lambda=0., bias=True, act='relu', norm=None):
        super(ClusterGcnConv, self).__init__()
        self.gconv = tg.nn.ClusterGCNConv(in_channels=in_channels, out_channels=out_channels, diag_lambda=diag_lambda, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, x_0, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out

# [0.1, 0.2, 0.3, 0.4, 0.5] [0.1]
alpha_const = 0.5 # (0.5) 
# [0.5, 1.0, 1.5] [0.4, 0.5, 0.6]
theta_const = 1.0 # (1.0) 0.5 


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge',
                 act='relu', norm=None, bias=True, heads=8, layer=1, dropout=0.):
        super(GraphConv, self).__init__()
        self.conv = conv.lower()
        if conv.lower() == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gat':
            self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads, dropout=dropout)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gin':
            self.gconv = GinConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'sage':
            self.gconv = SAGEConv(in_channels=in_channels, out_channels=out_channels, normalize=True, bias=bias, relative=False, norm=norm)
        elif conv.lower() == 'rsage':
            self.gconv = SAGEConv(in_channels=in_channels, out_channels=out_channels, normalize=True, bias=bias, relative=True, norm=norm)
        elif conv.lower() == 'sageii':
            self.gconv = SAGEConvII(in_channels=in_channels, out_channels=out_channels, normalize=True, bias=bias, relative=False,
                                    alpha=alpha_const, theta=theta_const, layer=layer, norm=norm)
        elif conv.lower() == 'rsageii':
            self.gconv = SAGEConvII(in_channels=in_channels, out_channels=out_channels, normalize=True, bias=bias, relative=True,
                                    alpha=alpha_const, theta=theta_const, layer=layer, norm=norm)
        elif conv.lower() == 'gcnii':
            self.gconv = GCNIIConv(in_channels, out_channels, act, norm, bias, layer, alpha=alpha_const, theta=theta_const)
        elif conv.lower() == 'ginii':
            self.gconv = GINConvII(channels=in_channels, eps=0., train_eps=False, alpha=alpha_const, theta=theta_const, layer=layer, norm=norm, bias=bias)
        elif conv.lower() == 'gatii':
            self.gconv = GATConvII(in_channels=in_channels, out_channels=out_channels//heads, heads=heads, concat=True,
                                   negative_slope=0.2, dropout=dropout, add_self_loops=True, bias=bias, alpha=alpha_const, theta=theta_const, layer=layer, norm=norm)
        elif conv.lower() == 'gatiia':
            self.gconv = GATConvIIa(in_channels=in_channels, out_channels=out_channels//heads, heads=heads, concat=True,
                                   negative_slope=0.2, dropout=dropout, add_self_loops=True, bias=bias,alpha=alpha_const, theta=theta_const, layer=layer, norm=norm)
        elif conv.lower() == 'appnp':
            self.gconv = APPNPConv(K=10, alpha=0.1, dropout=dropout)
        elif conv.lower() == 'clustergcn':
            self.gconv = ClusterGcnConv(in_channels, out_channels, diag_lambda=0., bias=bias, act=act, norm=norm)
        elif conv.lower() == 'clustergcnii':
            self.gconv = ClusterGCNConvII(in_channels, out_channels, diag_lambda=0., bias=bias, alpha=alpha_const, theta=theta_const, layer=layer, norm=norm)
        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))

    def forward(self, x, x_0, edge_index, training=None):
        return self.gconv(x, x_0, edge_index)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None):
        edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
        :input: (x0, x1, x2, ... , xi), batch
        :output:(x0, x1, x2, ... , xi ,xi+1) , batch
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None):
        return self.body(x, batch) + x*self.res_scale, batch


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(channels*2, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)

    def forward(self, x, batch=None):
        dense = self.body(x, batch)
        return torch.cat((x, dense), 1), batch


class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, channels,  conv='edge', act='relu', norm=None, bias=True, heads=8,  res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = GraphConv(channels, channels, conv, act, norm, bias, heads)
        self.res_scale = res_scale

    def forward(self, x, edge_index):
        return self.body(x, edge_index) + x*self.res_scale, edge_index


class DenseGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, in_channels,  out_channels, conv='edge', act='relu', norm=None, bias=True, heads=8):
        super(DenseGraphBlock, self).__init__()
        self.body = GraphConv(in_channels, out_channels, conv, act, norm, bias, heads)

    def forward(self, x, edge_index):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1), edge_index

