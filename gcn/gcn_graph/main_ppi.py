import __init__
import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
from gcn import utils
import logging
import argparse
import torch.nn as nn
from torch_geometric.datasets import PPI, Planetoid
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch.utils
import torch.backends.cudnn as cudnn
from model import NetworkPPI as Network
# this is used for loading cells for evaluation
import genotypes
import random
from sklearn.metrics import f1_score
import numpy

from torch_geometric.data import Batch, ClusterData, ClusterLoader, DataLoader

# from warmup_scheduler import GradualWarmupScheduler
# import adabound

from nfnets.agc import AGC # Needs testing

from torch_geometric.utils import dropout_adj
from torch_sparse.storage import SparseStorage
# edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]


parser = argparse.ArgumentParser("eval")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--phase', type=str, default='train', help='train/test')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.002, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=20, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--steplr_epochs', type=int, default=4000, help='steplr epochs')
parser.add_argument('--init_channels', type=int, default=512, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=5, help='total number of cells')
parser.add_argument('--model_path', type=str, default='log/ckpt', help='path to save the model / pretrained')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability') # 
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=123, help='random seed') # 123 42
parser.add_argument('--arch', type=str, default='Cri2_PPI_Best', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--in_channels', default=50, type=int, help='the channel size of input point cloud ')

parser.add_argument('--disable_scheduler', action='store_true', default=False, help='disable cosine annealing scheduler')
parser.add_argument('--fix_drop_path_prob', action='store_true', default=False, help='fix drop path probability')
parser.add_argument('--scheduler', type=str, default='cosine', help='cosine or step')
parser.add_argument('--warmup', action='store_true', default=False, help='lr warm up')
parser.add_argument('--patience', type=int, default=0, help='patience')
parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd or agc')

parser.add_argument('--cluster', action='store_true', default=False, help='use cluster gcn loader')
parser.add_argument('--cluster_sparse', action='store_true', default=False, help='sparse tensor for cluster gcn loader')
parser.add_argument('--dropedge', action='store_true', default=False, help='use dropedge')

parser.add_argument('--block', action='store_true', default=False, help='training block search')

parser.add_argument('--dataset', type=str, default='PPI', help='dataset: PPI, Cora')

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.save = 'log/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
        

from torch_sparse import SparseTensor
def ToSS(data):
    remove_edge_index = True
    fill_cache = True

    assert data.edge_index is not None

    (row, col), N, E = data.edge_index, data.num_nodes, data.num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    if remove_edge_index:
        data.edge_index = None

    value = None
    for key in ['edge_weight', 'edge_attr', 'edge_type']:
        if data[key] is not None:
            value = data[key][perm]
            if remove_edge_index:
                data[key] = None
            break

    for key, item in data:
        if item.size(0) == E:
            data[key] = item[perm]

    data.adj_t = SparseTensor(row=col, col=row, value=value,
                                sparse_sizes=(N, N), is_sorted=True)

    if fill_cache:  # Pre-process some important attributes.
        data.adj_t.storage.rowptr()
        data.adj_t.storage.csr2csc()

    return data


# def randomedge_sampler(edge_index, percent):
#     # nnz = self.train_adj.nnz
#     # perm = np.random.permutation(nnz)
#     # preserve_nnz = int(nnz*percent)
#     # perm = perm[:preserve_nnz]
#     # r_adj = sp.coo_matrix((self.train_adj.data[perm],
#     #                         (self.train_adj.row[perm],
#     #                         self.train_adj.col[perm])),
#     #                         shape=self.train_adj.shape)
#     nnz = edge_index.size(1)
#     preserve_nnz = int(nnz*percent)
#     edge_index=edge_index[:,torch.randperm(edge_index.size()[1])]
#     edge_index=edge_index[:,:preserve_nnz]
#     return edge_index
#     # adj_t = torch.sparse.FloatTensor(torch.stack((data.adj_t.storage.row(), data.adj_t.storage.col()), dim=0), 
#     #                                      data.adj_t.storage.value(), torch.Size([data.num_nodes, data.num_nodes]))


def train_cora():
    model.train()
    optimizer.zero_grad()
    out, _ = model(data, True)
    out = out.log_softmax(dim=-1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test_cora():
    model.eval()
    pred, _ = model(data, False)
    pred = pred.log_softmax(dim=-1).argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


def train():
    best_val_acc = 0.
    test_acc = 0.
    best_test_acc = 0.
    best_test_acc_mean = 0.
    train_acc = 0.
    best_epoch = 0
    test_acc_when_best_val_mean = 0.
    early_stop_counter = 0
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(args.epochs):
        if not args.disable_scheduler:
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        else:
            logging.info('epoch %d lr %e', epoch, args.learning_rate)
        # else:
        #     logging.info('epoch %d lr %e', epoch, optimizer.param_groups[0]['lr'])

        if not args.fix_drop_path_prob:
            model.drop_path_prob = args.drop_path_prob * epoch / (args.epochs)
            # model.drop_path_prob = args.drop_path_prob * epoch / (args.epochs*0.5)
            # if epoch >= (args.epochs*0.5):
            #     model.drop_path_prob = args.drop_path_prob
        else:
            model.drop_path_prob = args.drop_path_prob

        logging.info('drop prob %f', model.drop_path_prob)

        if args.dataset == 'PPI':
            train_acc, train_obj = train_step(train_queue, model, criterion, optimizer)
            valid_acc, valid_obj, val_acc_mean = infer(valid_queue, model, criterion, "val")
            test_acc, test_obj, test_acc_mean = infer(test_queue, model, criterion, "test")

            if valid_acc > best_val_acc:
                early_stop_counter = 0
                test_acc_when_best_val_mean = test_acc_mean
                best_epoch = epoch
                best_val_acc = valid_acc
                test_acc_when_best_val = test_acc
                utils.save(model, os.path.join(args.save, 'best_weights.pt'))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_mean = test_acc_mean
            # logging.info('best_epoch %d\ttrain_acc %f\tvalid_acc %f\tbest_val_acc %f\ttest_acc %f\tbest_test_acc %f\tfinal_best_test %f\n',
            #              best_epoch, train_acc, valid_acc, best_val_acc, test_acc, best_test_acc, test_acc_when_best_val)
            logging.info('best_epoch {}, {:.4f}, test_acc_when_best_val_mean {:.4f}, val_acc {:.4f}, val_acc_mean {:.4f}, best_val_acc {:.4f}\ntest_acc {:.4f}, test_acc_mean {:.4f}, best_test_acc {:.4f}, best_test_acc_mean {:.4f}, train_acc {:.4f}\n'.format(
                    best_epoch,test_acc_when_best_val,test_acc_when_best_val_mean,valid_acc,val_acc_mean,best_val_acc,test_acc,test_acc_mean,best_test_acc,best_test_acc_mean,train_acc))
        elif args.dataset == 'Cora':
            train_loss = train_cora()
            train_acc, val_acc, tmp_test_acc = test_cora()
            if val_acc > best_val_acc:
                early_stop_counter = 0
                best_epoch = epoch
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            logging.info(f'Epoch: {epoch:04d}, Loss: {train_loss:.4f} Train: {train_acc:.4f}, '
                f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
                f'Final Test: {test_acc:.4f}, best_epoch: {best_epoch:04d}\n')

        utils.save(model, os.path.join(args.save, 'weights.pt'))

        if not args.disable_scheduler:
            scheduler.step()

        early_stop_counter += 1
        if  args.patience > 0 and early_stop_counter > args.patience:
            break

    if args.dataset == 'PPI':
        logging.info(
            'Finish! best_val_acc %f\t test_class_acc_when_best %f \t best test %f',
            best_val_acc, test_acc_when_best_val, best_test_acc)

        logging.info('best_epoch {}, {:.4f}, test_acc_when_best_val_mean {:.4f}, best_val_acc {:.4f}, best_test_acc {:.4f}, best_test_acc_mean {:.4f}, train_acc {:.4f}'.format(
                    best_epoch,test_acc_when_best_val,test_acc_when_best_val_mean,best_val_acc,best_test_acc,best_test_acc_mean,train_acc))


def train_step(train_queue, model, criterion, optimizer):
    objs = utils.AverageMeter()
    micro_f1 = 0.
    count = 0.
    for step, input in enumerate(train_queue):
        if args.cluster_sparse:
            input = ToSS(input)
        model.train()
        input = input.to(DEVICE)
        target = input.y
        n = input.x.size(0)

        optimizer.zero_grad()
        logits, logits_aux = model(input, True)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        if args.optimizer != 'agc':
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        micro_f1 += utils.mF1(logits, target) * n
        count += n
        objs.update(loss.item(), n)
    micro_f1 = float(micro_f1) / count
    return micro_f1, objs.avg


def infer(valid_queue, model, criterion, mode):
    objs = utils.AverageMeter()
    model.eval()
    count = 0.
    micro_f1 = 0.
    micro_f1_arr = []
    ys, preds = [], []
    with torch.no_grad():
        for step, input in enumerate(valid_queue):
            if args.cluster_sparse:
                input = ToSS(input)
            ys.append(input.y)
            input = input.to(DEVICE)
            target = input.y
            
            if mode == "test": t_inference = time.time()
            logits, _ = model(input, False)
            if mode == "test": logging.info("Inference time: {:.4f}s".format(time.time() - t_inference))

            preds.append((logits > 0).float().cpu())
            loss = criterion(logits, target)

            n = target.size(0)
            f1 = utils.mF1(logits, target)
            micro_f1_arr.append(f1)
            # micro_f1 += f1 * n
            count += n
            objs.update(loss.item(), n)
    # micro_f1 = float(micro_f1) / count
    # logging.info("micro_f1_arr: {}".format(micro_f1_arr))
    # logging.info("micro_f1: {}".format(numpy.mean(micro_f1_arr)))
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    micro_f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return micro_f1, objs.avg, numpy.mean(micro_f1_arr)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # torch.cuda.set_per_process_memory_fraction(0.22)
    # torch.cuda.empty_cache()
    # total_memory = torch.cuda.get_device_properties(0).total_memory
    # print(total_memory)

    
    if args.dataset == 'PPI':
        if args.cluster:
            train_dataset = PPI(os.path.join(args.data, 'ppi'), split='train')
            valid_dataset = PPI(os.path.join(args.data, 'ppi'), split='val')
            test_dataset = PPI(os.path.join(args.data, 'ppi'), split='test')
            train_data = Batch.from_data_list(train_dataset)
            cluster_data = ClusterData(train_data, num_parts=50, recursive=False,
                                    save_dir=train_dataset.processed_dir)
            train_queue = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                                        num_workers=12)
            valid_queue = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            test_queue = DataLoader(test_dataset, batch_size=1, shuffle=False)
        else:
            train_dataset = PPI(os.path.join(args.data, 'ppi'), split='train', pre_transform=T.ToSparseTensor())
            valid_dataset = PPI(os.path.join(args.data, 'ppi'), split='val', pre_transform=T.ToSparseTensor())
            test_dataset = PPI(os.path.join(args.data, 'ppi'), split='test', pre_transform=T.ToSparseTensor())
            train_queue = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            valid_queue = DataLoader(valid_dataset, batch_size=1, shuffle=False)
            test_queue = DataLoader(test_dataset, batch_size=1, shuffle=False)
            if args.cluster:
                n_classes = train_dataset.num_classes
            else:
                n_classes = train_queue.dataset.num_classes
    elif args.dataset == 'Cora':
        dataset = 'Cora'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', dataset)
        transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        dataset = Planetoid(path, dataset, split='public', transform=transform) # public full
        data = dataset[0]
        data = data.to(DEVICE)
        n_classes = dataset.num_classes
        args.in_channels = dataset.num_features
        

    if args.block:
        genotype = args.arch
    else:
        genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, n_classes, args.num_cells, args.auxiliary, genotype,
                    in_channels=args.in_channels, dropedge=args.dropedge, cluster=args.cluster, cluster_sparse=args.cluster_sparse, block=args.block)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.BCEWithLogitsLoss().cuda()

    if args.dataset == 'PPI':
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        elif args.optimizer == 'agc':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['classifier']) # 
        else:
            # optimizer = adabound.AdaBound(model.parameters(), lr=args.learning_rate, final_lr=0.1)
            raise NotImplementedError('optimizer type [%s] is not found' % args.optimizer)
    elif args.dataset == 'Cora':
        optimizer = torch.optim.Adam([
            dict(params=model.cells.parameters(), weight_decay=0.01),
            dict(params=model.stem.parameters(), weight_decay=5e-4),
            dict(params=model.preprocess.parameters(), weight_decay=5e-4)
        ], lr=args.learning_rate)

    if not args.disable_scheduler:
        if args.scheduler == 'cosine':
            # scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        else:
            scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer, args.steplr_epochs, gamma=0.5)

        if args.warmup:
            # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100, after_scheduler=scheduler_)
            raise NotImplementedError('warm up is not implemented')
        else:
            scheduler = scheduler_

    if args.disable_scheduler and args.warmup:
        scheduler_ = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs, gamma=1.0)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100, after_scheduler=scheduler_)
        raise NotImplementedError('warm up is not implemented')
        args.disable_scheduler = False

    if args.phase == 'test':
        logging.info("===> Loading checkpoint '{}'".format(args.model_path))
        utils.load(model, args.model_path)
        test_acc, test_obj, test_acc_mean = infer(test_queue, model, criterion, "test")
        logging.info('Finish Testing! test_acc {:.4f} {:.4f}'.format(test_acc, test_acc_mean))
    else:
        t_total = time.time()
        train()
        logging.info("Train cost: {:.4f}s".format(time.time() - t_total))

