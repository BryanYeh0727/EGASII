import __init__
import os
import os.path as osp
import sys
import time
import glob
import math
import numpy as np
import torch
from gcn import utils
import logging
import argparse
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import PPI, Planetoid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.backends.cudnn as cudnn
import torch.distributions.categorical as cate
import torchvision.utils as vutils
from sklearn.metrics import f1_score

from model_search import Network
from architect import Architect
from tensorboardX import SummaryWriter
import random

import copy
from genotypes import PRIMITIVES

from nfnets.agc import AGC # Needs testing


# torch_geometric.set_debug(True)
parser = argparse.ArgumentParser("ppi")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=2, help='batch size') # 4 2 1
parser.add_argument('--batch_increase', default=0, type=int, help='how much does the batch size increase after making a decision')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels') # 
parser.add_argument('--num_cells', type=int, default=1, help='total number of cells')
parser.add_argument('--n_steps', type=int, default=3, help='total number of layers in one cell')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='PPI', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed') # 2 42
parser.add_argument('--random_seed', action='store_true', help='use seed randomly')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_dec_epoch', type=int, default=16, help='warmup decision epoch') # 16 9 80 40
parser.add_argument('--decision_freq', type=int, default=21, help='decision freq epoch') # 35   21 7 10 10
parser.add_argument('--history_size', type=int, default=4, help='number of stored epoch scores')
parser.add_argument('--use_history', action='store_true', help='use history for decision')
parser.add_argument('--in_channels', default=50, type=int, help='the channel size of input point cloud ')
parser.add_argument('--post_val', action='store_true', default=False, help='validate after each decision')

# parser.add_argument('--gcnii_dropout', type=float, default=0.2, help='gcnii dropout probability')
parser.add_argument('--gumbel', action='store_true', help='use gumbel softmax')
parser.add_argument('--disable_scheduler', action='store_true', default=False, help='disable cosine annealing scheduler')
parser.add_argument('--optimizer', type=str, default='sgd', help='adam or sgd or agc')
parser.add_argument('--temp_decay', type=float, default=0.923, help='temp decay')

parser.add_argument('--dataset', type=str, default='PPI', help='dataset: PPI, Cora')


args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.save = 'log/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(log_dir=args.save, max_queue=50)


def histogram_average(history, probs):
    histogram_inter = torch.zeros(probs.shape[0], dtype=torch.float).cuda()
    if not history:
        return histogram_inter
    for hist in history:
        histogram_inter += utils.histogram_intersection(hist, probs)
    histogram_inter /= len(history)
    return histogram_inter


def score_image(type, score, epoch):
    score_img = vutils.make_grid(
        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(score, 1), 2), 3),
        nrow=7,
        normalize=True,
        pad_value=0.5)
    writer.add_image(type + '_score', score_img, epoch)


def edge_decision(type, alphas, selected_idxs, candidate_flags, curstage_selected_idxs, curstage_candidate_flags, 
                  probs_history, epoch, model, args, switches, decide_edge, tau):
    # mat = F.gumbel_softmax(torch.stack(alphas, dim=0), dim=-1, tau=tau).detach()
    mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
    logging.info(mat)
    importance = torch.sum(mat[:, 1:], dim=-1)
    # logging.info(type + " importance {}".format(importance))

    probs = mat[:, 1:] / importance[:, None]
    # print(type + " probs", probs)
    entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.size()[1])
    # logging.info(type + " entropy {}".format(entropy))

    if args.use_history:  # SGAS Cri.2
        # logging.info(type + " probs history {}".format(probs_history))
        histogram_inter = histogram_average(probs_history, probs)
        # logging.info(type + " histogram intersection average {}".format(histogram_inter))
        probs_history.append(probs)
        if (len(probs_history) > args.history_size):
            probs_history.pop(0)

        score = utils.normalize(importance) * utils.normalize(
            1 - entropy) * utils.normalize(histogram_inter)
        # logging.info(type + " score {}".format(score))
    else:  # SGAS Cri.1
        score = utils.normalize(importance) * utils.normalize(1 - entropy)
        # logging.info(type + " score {}".format(score))

    if torch.sum(candidate_flags.int()) > 0 and \
            epoch >= args.warmup_dec_epoch and \
            (epoch - args.warmup_dec_epoch) % args.decision_freq == 0 and \
            decide_edge:
        masked_score = torch.min(score,
                                 (2 * candidate_flags.float() - 1) * np.inf) # 9 edges score
        selected_edge_idx = torch.argmax(masked_score)
        selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1  # add 1 since none op
        curstage_selected_idxs[selected_edge_idx] = selected_op_idx

        idxs = []
        for i in range(len(PRIMITIVES)):
            if switches[selected_edge_idx][i]:
                idxs.append(i)

        selected_idxs[selected_edge_idx] = idxs[selected_op_idx]

        candidate_flags[selected_edge_idx] = False
        curstage_candidate_flags[selected_edge_idx] = False
        alphas[selected_edge_idx].requires_grad = False
        if type == 'normal':
            reduction = False
        elif type == 'reduce':
            reduction = True
        else:
            raise Exception('Unknown Cell Type')
        # if 2 edge is decide for an intermediate node, 
        # change the candidate_flag of other edges to that intermediate node to False, 
        # and decide those edges to be zero operation
        candidate_flags, curstage_candidate_flags, selected_idxs = model.check_edges(candidate_flags, 
                                                                                     curstage_candidate_flags,
                                                                                     selected_idxs)
        logging.info("#" * 30 + " Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}, added edge {} with op idx {}".format(epoch,
                                                                                          type,
                                                                                          selected_idxs,
                                                                                          selected_edge_idx,
                                                                                          selected_op_idx))
        logging.info(type + "_candidate_flags {}".format(candidate_flags))
        score_image(type, score, epoch)
        return True, selected_idxs, candidate_flags, curstage_selected_idxs, curstage_candidate_flags

    else:
        logging.info("#" * 30 + " Not a Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}".format(epoch,
                                                            type,
                                                            selected_idxs))
        logging.info(type + "_candidate_flags {}".format(candidate_flags))
        score_image(type, score, epoch)
        return False, selected_idxs, candidate_flags, curstage_selected_idxs, curstage_candidate_flags


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.random_seed:
        args.seed = np.random.randint(0, 1000, 1)

    random.seed(args.seed[0])
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.dataset == 'PPI':
        # dataset ppi
        # pre_transform = T.Compose([T.GCNNorm(), T.ToSparseTensor()])
        train_dataset = PPI(os.path.join(args.data, 'ppi'), split='train', pre_transform=T.ToSparseTensor())
        valid_dataset = PPI(os.path.join(args.data, 'ppi'), split='val', pre_transform=T.ToSparseTensor())
        # train_dataset = GeoData.PPI(os.path.join(args.data, 'ppi'), split='train')
        train_queue = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # valid_dataset = GeoData.PPI(os.path.join(args.data, 'ppi'), split='val')
        valid_queue = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
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


    switches = []
    for i in range(9): # for 3 step, there are 9 edges in a cell
        switches.append([True for j in range(len(PRIMITIVES))]) # 10 primitives

    normal_selected_idxs = torch.tensor(9 * [-1], requires_grad=False, dtype=torch.int).cuda()
    normal_candidate_flags = torch.tensor(9 * [True], requires_grad=False, dtype=torch.bool).cuda()

    # num_to_drop = [3, 3, 4]
    num_to_drop = [2, 2, 2] # primitives to drop after each epoch
    # add_cells = [1, 3, 5]
    add_cells = [0, 1, 2] # cells added at each epoch
    # drop_rate = [0.0, 0.0, 0.0]
    # add_init_channels = [0, 16, 32]
    add_init_channels = [0, 0, 0]
    for sp in range(len(num_to_drop)): # 3 search stage
        decide_edge = True
        # if sp == 0:
        #     decide_edge = False
        # else:
        #     decide_edge = True

        curstage_selected_idxs = torch.tensor(9 * [-1], requires_grad=False, dtype=torch.int).cuda()
        curstage_candidate_flags = torch.tensor(9 * [True], requires_grad=False, dtype=torch.bool).cuda()
        switch_on = len(PRIMITIVES)
        if(sp):
            for i in range(sp):
                switch_on -= num_to_drop[i]

        criterion = torch.nn.BCEWithLogitsLoss().cuda()
        model = Network(args.init_channels + add_init_channels[sp], n_classes, args.num_cells + add_cells[sp], criterion,
                        steps=args.n_steps, in_channels=args.in_channels, switches=switches, 
                        normal_selected_idxs=normal_selected_idxs, normal_candidate_flags=normal_candidate_flags, 
                        curstage_selected_idxs=curstage_selected_idxs, curstage_candidate_flags=curstage_candidate_flags, switch_on=switch_on,
                        gumbel=args.gumbel, temp_decay=args.temp_decay).cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        num_edges = model._steps * 2 # 6
        post_train = 5
        # there will be a total of 6 times of edge decision
        # args.epochs = args.warmup_dec_epoch + args.decision_freq * (num_edges - 1) + post_train + 1 # 9+7*(6-1)+5+1=50
        # 2 edge decision for every search stage
        args.epochs = args.warmup_dec_epoch + args.decision_freq * (2 - 1) + post_train + 1 # 9+7*(2-1)+5+1=22
        # epoch 0~21
        # decision happens in epoch 9 and epoch 16
        logging.info("total epochs: %d", args.epochs)

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
                optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['classifier'])
            # optimizer = torch.optim.SGD(
            #     model.parameters(),
            #     args.learning_rate,
            #     momentum=args.momentum,
            #     weight_decay=args.weight_decay)
        elif args.dataset == 'Cora':
            optimizer = torch.optim.Adam([
                dict(params=model.cells.parameters(), weight_decay=0.01),
                dict(params=model.stem.parameters(), weight_decay=5e-4),
                dict(params=model.preprocess.parameters(), weight_decay=5e-4)
            ], lr=args.learning_rate)

        if not args.disable_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        architect = Architect(model, args)

        # normal_selected_idxs = torch.tensor(len(model.alphas_normal) * [-1], requires_grad=False, dtype=torch.int).cuda()
        # normal_candidate_flags = torch.tensor(len(model.alphas_normal) * [True], requires_grad=False, dtype=torch.bool).cuda()

        logging.info('normal_selected_idxs: {}'.format(model.normal_selected_idxs))
        logging.info('normal_candidate_flags: {}'.format(model.normal_candidate_flags))
        # model.normal_selected_idxs = normal_selected_idxs
        # model.normal_candidate_flags = normal_candidate_flags

        logging.info(F.softmax(torch.stack(model.alphas_normal, dim=0), dim=-1).detach())

        count = 0
        normal_probs_history = []
        train_losses, valid_losses = utils.AverageMeter(), utils.AverageMeter()
        for epoch in range(args.epochs):
            # lr = optimizer.param_groups[0]['lr']
            # logging.info('epoch %d lr %e', epoch, lr)
            if not args.disable_scheduler:
                logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            else:
                logging.info('epoch %d lr %e', epoch, args.learning_rate)

            if args.gumbel:
                logging.info('temperature:  %f', model.temperature)

            if args.dataset == 'PPI':
                # training
                train_acc, train_losses = train(train_queue, valid_queue, model, architect, criterion, optimizer, args.learning_rate, train_losses)
                valid_acc, valid_losses = infer(valid_queue, model, criterion, valid_losses)

            logging.info('train_acc %f\tvalid_acc %f', train_acc, valid_acc)

            # make edge decisions
            saved_memory_normal, model.normal_selected_idxs, \
            model.normal_candidate_flags, model.curstage_selected_idxs, \
            model.curstage_candidate_flags = edge_decision('normal',
                                                            model.alphas_normal,
                                                            model.normal_selected_idxs,
                                                            model.normal_candidate_flags,
                                                            model.curstage_selected_idxs,
                                                            model.curstage_candidate_flags,
                                                            normal_probs_history,
                                                            epoch,
                                                            model,
                                                            args,
                                                            switches,
                                                            decide_edge,
                                                            model.temperature)

            logging.info('curstage_selected_idxs: {}'.format(model.curstage_selected_idxs))
            logging.info('curstage_candidate_flags: {}'.format(model.curstage_candidate_flags))

            if saved_memory_normal:
                del train_queue, valid_queue
                torch.cuda.empty_cache()

                count += 1
                new_batch_size = args.batch_size + args.batch_increase * count
                logging.info("new_batch_size = {}".format(new_batch_size))

                train_queue = DataLoader(train_dataset, batch_size=new_batch_size, shuffle=True)
                valid_queue = DataLoader(valid_dataset, batch_size=new_batch_size, shuffle=False)

                if args.post_val:
                    valid_acc, valid_obj = infer(valid_queue, model, criterion)
                    logging.info('post valid_acc %f', valid_acc)

            writer.add_scalar('stats/train_acc', train_acc, epoch)
            writer.add_scalar('stats/valid_acc', valid_acc, epoch)
            utils.save(model, os.path.join(args.save, 'weights.pt'))

            model.temperature *= model.temp_decay

            if not args.disable_scheduler:
                scheduler.step()

        normal_selected_idxs = model.normal_selected_idxs
        normal_candidate_flags = model.normal_candidate_flags
        normal_prob = F.softmax(torch.stack(model.alphas_normal, dim=0), dim=-1).detach()
        # normal_prob = F.softmax(model.alphas_normal, dim=-1).data.cpu().numpy()
        for i in range(9):
            if model.normal_candidate_flags[i] == True: # if the edge is undecided
                idxs = []
                for j in range(len(PRIMITIVES)):
                    if switches[i][j]:
                        idxs.append(j)
                if sp == len(num_to_drop) - 1:
                    # for the last stage, drop all Zero operations
                    drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
                else:
                    # drop lowest architechture weight except Zero operation
                    drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
                for idx in drop:
                    logging.info(str(i) + ": " + str(idx) + " " + str(idxs[idx]))
                    switches[i][idxs[idx]] = False


    logging.info("#" * 30 + " Done " + "#" * 30)
    logging.info('genotype = %s', model.get_genotype())


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    input = input[1:]
    # input = input[1:-2]
    index = []
    for i in range(k):
        idx = torch.argmin(input)
        input[idx] = 1
        index.append(idx+1)
    return index

def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = torch.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, train_losses):
    micro_f1 = 0.
    count = 0.
    train_losses.reset()

    for step, input in enumerate(train_queue):
        model.train()
        input = input.to(DEVICE)
        target = input.y
        n = input.x.size(0)

        input_search = next(iter(valid_queue))
        input_search = input_search.to(DEVICE)
        target_search = input_search.y

        # unrolled False
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, training=True)

        optimizer.zero_grad()
        logits = model(input, True)
        loss = criterion(logits, target)

        loss.backward()
        if args.optimizer != 'agc':
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        micro_f1 += utils.mF1(logits, target) * n
        count += n
        train_losses.update(loss.item(), n)
    micro_f1 = float(micro_f1) / count
    return micro_f1, train_losses


def infer(valid_queue, model, criterion, valid_losses):
    model.eval()
    count = 0.
    micro_f1 = 0.
    valid_losses.reset()
    micro_f1_arr = []
    ys, preds = [], []
    with torch.no_grad():
      for step, input in enumerate(valid_queue):
          ys.append(input.y)
          input = input.to(DEVICE)
          target = input.y
          logits = model(input, False)
          preds.append((logits > 0).float().cpu())
          loss = criterion(logits, target)

          n = target.size(0)
        #   micro_f1 += utils.mF1(logits, target) * n
          f1 = f1_score(target.cpu().detach().numpy(), (logits > 0).cpu().detach().numpy(), average='micro')
          micro_f1_arr.append(f1)
          count += n
          valid_losses.update(loss.item(), n)
    # micro_f1 = float(micro_f1) / count
    logging.info("micro_f1_arr {}".format(micro_f1_arr))
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    micro_f1 = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
    return micro_f1, valid_losses

# def train_cora(data, model, architect, criterion, optimizer, lr, train_losses):
#     train_losses.reset()
#     model.train()
#     architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, training=True)
#     optimizer.zero_grad()
#     out, _ = model(data, True)
#     out = out.log_softmax(dim=-1)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return float(loss)

# @torch.no_grad()
# def test_cora():
#     model.eval()
#     pred, _ = model(data, False)
#     pred = pred.log_softmax(dim=-1).argmax(dim=-1)
#     accs = []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
#     return accs


if __name__ == '__main__':
    main()
