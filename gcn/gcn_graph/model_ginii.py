import __init__
import torch
from operations import *
import torch.nn.functional as F
from torch_sparse.storage import SparseStorage
from torch_sparse import SparseTensor
import torch_geometric as tg

norm = 'batch'
# norm = 'layer'

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, layer):
        super(Cell, self).__init__()
        # self.preprocess0 = MLP([C_prev_prev, C], 'relu', norm, bias=False)
        self.preprocess1 = MLP([C_prev, C*4], 'relu', norm, bias=False)
        self.layer = layer

        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        op = OPS["ginii"](C*4, 1, self.layer)
        self._ops += [op]

        self._indices = indices

    def forward(self, s0, s1, edge_index, drop_prob, x_0, training):
        # s0 = self.preprocess0(s0)
        # s0 = F.dropout(s0, p=drop_prob, training=self.training)
        s1 = self.preprocess1(s1)
        # s1 = F.dropout(s1, p=drop_prob, training=self.training)

        op1 = self._ops[0]
        h1 = op1(s1, x_0, edge_index, training)
        # if self.training and drop_prob > 0.:
        #     h1 = drop_path(h1, drop_prob)
        # h1 = F.dropout(h1, 0.2 ,training=self.training)
        return h1


class AuxiliaryHeadPPI(nn.Module):

    def __init__(self, C, num_classes):
        super(AuxiliaryHeadPPI, self).__init__()
        self.features = nn.Sequential(
            MLP([C, 128, 768], 'relu', norm, bias=False)

        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def randomedge_sampler(adj_t, percent):
    nnz = adj_t.nnz()
    # print(nnz)
    # print(adj_t.storage._sparse_sizes)
    perm = torch.randperm(nnz)
    perm = perm[:int(nnz*percent)]
    data_adj_t = SparseTensor(row=adj_t.storage.row()[perm], col=adj_t.storage.col()[perm], value=None,
                                sparse_sizes=adj_t.storage._sparse_sizes, 
                                is_sorted=False)
    # print(data_adj_t.storage._sparse_sizes)
    data_adj_t.storage.rowptr()
    data_adj_t.storage.csr2csc()
    return data_adj_t
    # return adj_t.from_storage(SparseStorage(row=adj_t.storage.row()[perm].contiguous(), rowptr=adj_t.storage._rowptr, col=adj_t.storage.col()[perm].contiguous(),
    #                                         value=None, sparse_sizes=adj_t.storage._sparse_sizes,
    #                                         rowcount=adj_t.storage._rowcount, colptr=adj_t.storage._colptr,
    #                                         colcount=adj_t.storage._colcount, csr2csc=adj_t.storage._csr2csc,
    #                                         csc2csr=adj_t.storage._csc2csr, is_sorted=True))

class NetworkPPI(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, stem_multiplier=3, in_channels=3, dropedge=False, cluster=False, cluster_sparse=False, block=False):
        super(NetworkPPI, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._in_channels = in_channels
        self.drop_path_prob = 0.
        self.dropedge = dropedge
        self.cluster = cluster
        self.cluster_sparse = cluster_sparse

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            MLP([in_channels, C_curr], None, norm, bias=False),
        )
        self.preprocess = MLP([in_channels, C*4], 'relu', norm, bias=False)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            if block:
                geno = eval("genotypes_block.%s" % (eval("genotypes_block.%s" % genotype))[i])
                cell = Cell(geno, C_prev_prev, C_prev, C_curr, i+1)
            else:
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr, i+1)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadPPI(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev + 1, num_classes)
        # self.gconv = tg.nn.APPNP(K=2, alpha=0.1)

    def forward(self, input, training):
        logits_aux = None
        if self.cluster and not self.cluster_sparse:
            x, edge_index = input.x, input.edge_index
        else:
            x, edge_index = input.x, input.adj_t

        if self.training and self.dropedge:
            edge_index = randomedge_sampler(edge_index, 0.95)

        s0 = s1 = self.stem(x)
        x_0 = self.preprocess(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, edge_index, self.drop_path_prob, x_0, training)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1.unsqueeze(0)).squeeze(0)
        logits = self.classifier(torch.cat((out, s1), dim=1))

        # logits = self.gconv(logits, edge_index)
        
        return logits, logits_aux


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

