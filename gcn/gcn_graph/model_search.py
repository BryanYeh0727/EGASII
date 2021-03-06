import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS, MLP
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np

from genotypes import PRIMITIVES


norm = 'batch'
# norm = 'layer'


class MixedOp(nn.Module):

    def __init__(self, C, stride, switch, flag, selected_idx, layer):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()

        if flag:
            for i in range(len(switch)):
                if switch[i]:
                    primitive = PRIMITIVES[i]
                    op = OPS[primitive](C, stride, layer)
                    self._ops.append(op)
            # for primitive in PRIMITIVES:
            #     op = OPS[primitive](C, stride, False)
            #     self._ops.append(op)
        else:
            primitive = PRIMITIVES[selected_idx]
            op = OPS[primitive](C, stride, layer)
            self._ops.append(op)

    def forward(self, x, edge_index, weights, selected_idx, x_0, training):
        if selected_idx is None:
            return sum(w * op(x, x_0, edge_index, training) for w, op in zip(weights, self._ops))
        else:  # unchosen operations are pruned
            return self._ops[selected_idx](x, x_0, edge_index, training)

        # x_ = x
        # if selected_idx is None:
        #     return sum(w * (x_ + op(x, x_0, edge_index, training)) for w, op in zip(weights, self._ops))
        # else:  # unchosen operations are pruned
        #     return x_ + self._ops[selected_idx](x, x_0, edge_index, training)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, switches, normal_candidate_flags, normal_selected_idxs, layer):
        super(Cell, self).__init__()
        # self.preprocess = MLP([3*C, C], 'relu', 'batch', bias=False)
        self.preprocess0 = MLP([C_prev_prev, C], 'relu', norm, bias=False)
        self.preprocess1 = MLP([C_prev, C], 'relu', norm, bias=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        num_of_edge = 0
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride, switches[num_of_edge], normal_candidate_flags[num_of_edge], normal_selected_idxs[num_of_edge], layer)
                num_of_edge = num_of_edge + 1
                self._ops.append(op)

    def forward(self, s0, s1, edge_index, weights, curstage_selected_idxs, curstage_candidate_flags, selected_idxs, x_0, training):
        # x_0 = self.preprocess(x_0)
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            o_list = []
            for j, h in enumerate(states):
                if selected_idxs[offset + j] == -1: # undecided mix edges
                    o = self._ops[offset + j](h, edge_index, weights[offset + j], None, x_0, training)  # call the gcn module
                    o_list.append(o)
                elif selected_idxs[offset + j] == PRIMITIVES.index('none'): # pruned edges
                    # pass
                    continue
                else:  # decided discrete edges
                    if curstage_candidate_flags[offset + j]: # if the edge is not decided on this stage
                        o = self._ops[offset + j](h, edge_index, None, 0, x_0, training)
                    else:
                        o = self._ops[offset + j](h, edge_index, None, curstage_selected_idxs[offset + j], x_0, training)
                    # o = self._ops[offset + j](h, edge_index, None, selected_idxs[offset + j])
                    o_list.append(o)
            s = sum(o_list)
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, num_cells, criterion, 
                 normal_selected_idxs, normal_candidate_flags, curstage_selected_idxs, curstage_candidate_flags, 
                 steps=4, multiplier=4, stem_multiplier=3, in_channels=3, switches=[], switch_on=8, gumbel=False, temp_decay=0.923):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._num_cells = num_cells
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._in_channels = in_channels
        self.gumbel = gumbel
        self.temperature = 5.0
        self.temp_decay = temp_decay # (0.965 if 90 epochs) (0.938 if 50 epochs) (0.923 if 40 epochs)

        self.normal_selected_idxs = normal_selected_idxs
        self.normal_candidate_flags = normal_candidate_flags
        self.curstage_selected_idxs = curstage_selected_idxs
        self.curstage_candidate_flags = curstage_candidate_flags

        self.switches = switches
        # ons = 0
        # for i in range(len(switches[0])):
        #     if switches[0][i]:
        #         ons = ons + 1
        self.switch_on = switch_on # how many switches are on for one edge


        # stem_multiplier = 1
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            MLP([in_channels, C_curr], None, norm, bias=False),
        )
        self.preprocess = MLP([in_channels, C], 'relu', norm, bias=False)


        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(self._num_cells):
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, switches, normal_candidate_flags, normal_selected_idxs, i+1)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev + 1, num_classes)

        self._initialize_alphas()

        # self.normal_selected_idxs = torch.tensor(len(self.alphas_normal) * [-1], requires_grad=False, dtype=torch.int)
        # self.normal_candidate_flags = torch.tensor(len(self.alphas_normal) * [True],
        #                                            requires_grad=False, dtype=torch.bool)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._num_cells, self._criterion, self._steps,
                            in_channels=self._in_channels).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        model_new.normal_selected_idxs = self.normal_selected_idxs
        return model_new

    def forward(self, input, training):
        x, edge_index = input.x, input.adj_t
        s0 = s1 = self.stem(x)
        x_0 = self.preprocess(x)
        for i, cell in enumerate(self.cells):
            weights = []
            n = 2
            start = 0
            for _ in range(self._steps):
                end = start + n
                for j in range(start, end):
                    if self.gumbel:
                        weights.append(nn.functional.gumbel_softmax(self.alphas_normal[j], tau=self.temperature, dim=-1))
                    else:
                        weights.append(F.softmax(self.alphas_normal[j], dim=-1))
                start = end
                n += 1

            selected_idxs = self.normal_selected_idxs
            s0, s1 = s1, cell(s0, s1, edge_index, weights, self.curstage_selected_idxs, self.curstage_candidate_flags, selected_idxs, x_0, training)
        out = self.global_pooling(s1.unsqueeze(0)).squeeze(0)
        logits = self.classifier(torch.cat((out, s1), dim=1))
        return logits

    def _loss(self, input, target, training):
        logits = self(input, training)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # num_ops = len(PRIMITIVES)
        num_ops = self.switch_on
        self.alphas_normal = []
        for i in range(self._steps):
            for n in range(2 + i):
                self.alphas_normal.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
        self._arch_parameters = [
            self.alphas_normal
        ]

    def arch_parameters(self):
        return self.alphas_normal # dimension: [9, switch_on]

    def check_edges(self, flags, curstage_flags, selected_idxs):
        n = 2
        max_num_edges = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            if num_selected_edges >= max_num_edges:
                for j in range(start, end):
                    if flags[j]:
                        flags[j] = False
                        curstage_flags[j] = False
                        selected_idxs[j] = PRIMITIVES.index('none') # pruned edges
                        self.alphas_normal[j].requires_grad = False
                    else:
                        pass
            start = end
            n += 1

        return flags, curstage_flags, selected_idxs

    def parse_gene(self, selected_idxs):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    raise Exception("Contain undecided edges")
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def parse_gene_force(self, flags, selected_idxs, alphas):
        gene = []
        n = 2
        max_num_edges = 2
        start = 0
        mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
        importance = torch.sum(mat[:, 1:], dim=-1)
        masked_importance = torch.min(importance, (2 * flags.float() - 1) * np.inf)
        for _ in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            num_edges_to_select = max_num_edges - num_selected_edges
            if num_edges_to_select > 0:
                post_select_edges = torch.topk(masked_importance[start: end], k=num_edges_to_select).indices + start
            else:
                post_select_edges = []
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    if num_edges_to_select <= 0:
                        raise Exception("Unknown errors")
                    else:
                        if j in post_select_edges:
                            idx = torch.argmax(alphas[j][1:]) + 1
                            gene.append((PRIMITIVES[idx], j - start))
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def get_genotype(self, force=False):
        if force:
            gene_normal = self.parse_gene_force(self.normal_candidate_flags,
                                                self.normal_selected_idxs,
                                                self.alphas_normal)
        else:
            gene_normal = self.parse_gene(self.normal_selected_idxs)
        n = 2
        concat = range(n + self._steps - self._multiplier, self._steps + n)
        genotype = Genotype(normal=gene_normal, normal_concat=concat)
        return genotype
