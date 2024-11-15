# from asyncore import write
# import imp
import os
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.VNet import s4GAN_discriminator_ECSA_3d
from networks.net_factory import net_factory
from utils.utils import context_mask_Pancreas

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data_split/Pancreas', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='AstMatch', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=62, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.05, help='maximum epoch number to train')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96], help='patch size of network input')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=12, help='trained samples')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
parser.add_argument('--D_lr', type=float,  default=1e-4, help='maximum discriminator learning rate to train')
parser.add_argument("--threshold_st", type=float, default=0.6, help="threshold_st for the self-training threshold.")
parser.add_argument("--lambda-fm", type=float, default=0.1, help="lambda_fm for feature-matching loss.")
parser.add_argument("--lambda-st", type=float, default=1.0, help="lambda_st for self-training.")
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')

parser.add_argument('--num_augs', default=6, type=int, help='num_augs')
parser.add_argument('--flag_use_random_num_sampling', type=str, default=True, help='flag_use_random_num_sampling')

args = parser.parse_args()

def find_good_bad_maps(D1_out_weak, outputs_weak, outputs_strong_1, outputs_strong_2, weak_inputs, strong_inputs, label, args):
    labeled_img = weak_inputs[:args.labeled_bs]
    unlabeled_img_strong1 = strong_inputs[:args.labeled_bs]
    unlabeled_img_strong2 = strong_inputs[args.labeled_bs:]

    label=label.squeeze(dim=1)

    good_count = 0
    bad_count = 0
    for i in range(D1_out_weak.size(0)):
        if D1_out_weak[i] > args.threshold_st:
            good_count +=1
        else:
            bad_count+=1

    if good_count > 0:
        print ('Above ST-Threshold : ', good_count, '/', args.labeled_bs)
        pred_sel_strong_1 = torch.Tensor(good_count, outputs_weak.size(1), outputs_weak.size(2), outputs_weak.size(3), outputs_weak.size(4))
        pred_sel_strong_2 = torch.Tensor(good_count, outputs_weak.size(1), outputs_weak.size(2), outputs_weak.size(3), outputs_weak.size(4))
        label_sel = torch.Tensor(good_count, outputs_weak.size(2), outputs_weak.size(3), outputs_weak.size(4))
        num_sel = 0
        for j in range(D1_out_weak.size(0)):
            if D1_out_weak[j] > args.threshold_st:
                pred_sel_strong_1[num_sel] = outputs_strong_1[j]
                pred_sel_strong_2[num_sel] = outputs_strong_2[j]
                label_sel[num_sel] = getLargestCC(outputs_weak[j].argmax(dim=0))
                num_sel +=1
    else:
        pred_sel_strong_1=0
        pred_sel_strong_2=0
        label_sel=0

    if bad_count >0:
        print ('Below ST-Threshold : ', bad_count, '/', args.labeled_bs)
        mask_ratio = 0.3 ** (1 / 3)
        img_mask, _ = context_mask_Pancreas(labeled_img, mask_ratio)
        mixed_img_strong_1 = torch.Tensor(bad_count, labeled_img.size(1), labeled_img.size(2), labeled_img.size(3), labeled_img.size(4))
        mixed_img_strong_2 = torch.Tensor(bad_count, labeled_img.size(1), labeled_img.size(2), labeled_img.size(3), labeled_img.size(4))
        mixed_label = torch.Tensor(bad_count, labeled_img.size(2), labeled_img.size(3), labeled_img.size(4))
        num_sel = 0
        for k in range(D1_out_weak.size(0)):
            if D1_out_weak[k] <= args.threshold_st:
                choice = random.randint(0, 1)
                if choice==0:
                    mixed_img_strong_1[num_sel] = labeled_img[k] * img_mask + unlabeled_img_strong1[k] * (1 - img_mask)
                    mixed_img_strong_2[num_sel] = labeled_img[k] * img_mask + unlabeled_img_strong2[k] * (1 - img_mask)
                    mixed_label[num_sel] = label[k] * img_mask + getLargestCC(outputs_weak[k].argmax(dim=0)) * (1 - img_mask)
                    num_sel += 1
                else:
                    mixed_img_strong_1[num_sel] = labeled_img[k] * (1-img_mask) + unlabeled_img_strong1[k] * img_mask
                    mixed_img_strong_2[num_sel] = labeled_img[k] * (1-img_mask) + unlabeled_img_strong2[k] * img_mask
                    mixed_label[num_sel] = label[k] * (1-img_mask) + getLargestCC(outputs_weak[k].argmax(dim=0)) * img_mask
                    num_sel += 1

    else:
        mixed_img_strong_1=0
        mixed_img_strong_2=0
        mixed_label=0

    if good_count==args.labeled_bs:
        return  pred_sel_strong_1.cuda(),pred_sel_strong_2.cuda(), label_sel.cuda(), 0, 0, 0, good_count, bad_count
    elif good_count==0:
        return 0, 0, 0, mixed_img_strong_1.cuda(), mixed_img_strong_2.cuda(), mixed_label.cuda(), good_count, bad_count
    else:
        return  pred_sel_strong_1.cuda(),pred_sel_strong_2.cuda(), label_sel.cuda(), mixed_img_strong_1.cuda(), mixed_img_strong_2.cuda(), mixed_label.cuda(), good_count, bad_count



def getLargestCC(segmentation):
    n_prob = segmentation.detach().cpu().numpy()
    labels = label(n_prob)
    if labels.max() != 0:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    else:
        largestCC = n_prob
    return torch.Tensor(largestCC).cuda()


def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For every value v in `labels`, the value in the output will be either 1 or 0. Each vector along the `dim`-th
    dimension has the "one-hot" format, i.e., it has a total length of `num_classes`,
    with a one and `num_class-1` zeros.
    Note that this will include the background label, thus a binary mask should be treated as having two classes.

    Args:
        labels: input tensor of integers to be converted into the 'one-hot' format. Internally `labels` will be
            converted into integers `labels.long()`.
        num_classes: number of output channels, the corresponding length of `labels[dim]` will be converted to
            `num_classes` from `1`.
        dtype: the data type of the output one_hot label.
        dim: the dimension to be converted to `num_classes` channels from `1` channel, should be non-negative number.

    Example:

    For a tensor `labels` of dimensions [B]1[spatial_dims], return a tensor of dimensions `[B]N[spatial_dims]`
    when `num_classes=N` number of classes and `dim=1`.

    .. code-block:: python

        from monai.networks.utils import one_hot
        import torch

        a = torch.randint(0, 2, size=(1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=0)
        print(out.shape)  # torch.Size([2, 2, 2, 2])

        a = torch.randint(0, 2, size=(2, 1, 2, 2, 2))
        out = one_hot(a, num_classes=2, dim=1)
        print(out.shape)  # torch.Size([2, 2, 2, 2, 2])

    """

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, epoch, max_epoch, lr):
    lr = lr_poly(lr, epoch, max_epoch, 0.9)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def self_train(args, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes, mode="train")

    model_D = s4GAN_discriminator_ECSA_3d(num_classes=args.num_classes).cuda()
    model_D2 = s4GAN_discriminator_ECSA_3d(num_classes=args.num_classes).cuda()

    db_train = Pancreas(base_dir=args.root_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(args.patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0005)
    Dopt = optim.Adam(model_D.parameters(), lr=args.D_lr, betas=(0.9,0.99), weight_decay=0.0005)
    Dopt_2 = optim.Adam(model_D2.parameters(), lr=args.D_lr, betas=(0.9,0.99), weight_decay=0.0005)

    Dice_Funtion=losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = args.self_max_iteration // len(trainloader) + 1
    lr_ = args.base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            now_lr_G = adjust_learning_rate(optimizer, iter_num, args.self_max_iteration, args.base_lr)
            now_lr_D = adjust_learning_rate(Dopt, iter_num, args.self_max_iteration, args.D_lr)
            now_lr_D2 = adjust_learning_rate(Dopt_2, iter_num, args.self_max_iteration, args.D_lr)

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            for param in model.parameters():
                param.requires_grad = True
            for param in model_D.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            model.train()
            model_D.eval()
            model_D2.eval()

            weak_inputs = volume_batch
            strong_inputs = torch.zeros(unlabeled_volume_batch.shape).cuda(non_blocking=True)
            strong_inputs = strong_inputs.repeat(2, 1, 1, 1, 1)
            for oo in range(unlabeled_volume_batch.shape[0]):
                cut_ratio_1 = random.uniform(0.3, 0.7)
                Cutout_opt_1 = losses.Cutout(5, int(40 * cut_ratio_1))
                RandAugment = losses.build_additional_strong_transform(args)
                strong_inputs[oo, ...] = Cutout_opt_1(RandAugment(unlabeled_volume_batch[oo, ...].unsqueeze(dim=0))).squeeze(dim=0)

                cut_ratio_2 = random.uniform(0.3, 0.7)
                Cutout_opt_2 = losses.Cutout(5, int(40 * cut_ratio_1))
                RandAugment_2 = losses.build_additional_strong_transform(args)
                strong_inputs[(oo+unlabeled_volume_batch.shape[0]), ...] = Cutout_opt_2(RandAugment_2(unlabeled_volume_batch[oo, ...].unsqueeze(dim=0))).squeeze(dim=0)

            outputs, _ = model(weak_inputs)
            outputs_strong, _ = model(strong_inputs)
            outputs_label, outputs_weak = outputs.chunk(2)
            outputs_strong_1, outputs_strong_2=outputs_strong.chunk(2)

            loss_seg = F.cross_entropy(outputs_label, label_batch[:args.labeled_bs].long(), reduction='mean')
            loss_seg_dice = Dice_Funtion(outputs_label, label_batch[:args.labeled_bs].unsqueeze(dim=1))
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            unlabeled_bs=args.batch_size-args.labeled_bs
            weak_netD = (volume_batch[args.labeled_bs:] - torch.min(volume_batch[args.labeled_bs:])) / (torch.max(volume_batch[args.labeled_bs:]) - torch.min(volume_batch[args.labeled_bs:]))
            strong_netD_s1 = (strong_inputs[:unlabeled_bs]-torch.min(strong_inputs[:unlabeled_bs]))/(torch.max(strong_inputs[:unlabeled_bs])-torch.min(strong_inputs[:unlabeled_bs]))
            strong_netD_s2 = (strong_inputs[unlabeled_bs:]-torch.min(strong_inputs[unlabeled_bs:]))/(torch.max(strong_inputs[unlabeled_bs:])-torch.min(strong_inputs[unlabeled_bs:]))

            pred_cat_weak = torch.cat((F.softmax(outputs_weak, dim=1), weak_netD), dim=1)
            pred_cat_weak_detach = torch.cat((F.softmax(outputs_weak, dim=1), weak_netD), dim=1).detach()
            pred_cat_strong_1 = torch.cat((F.softmax(outputs_strong_1, dim=1), strong_netD_s1), dim=1)
            pred_cat_strong_2 = torch.cat((F.softmax(outputs_strong_2, dim=1), strong_netD_s2), dim=1)

            D1_out_weak, D1_out_map_weak = model_D(pred_cat_weak)
            D2_out_weak, D2_out_map_weak = model_D2(pred_cat_weak_detach)
            D2_out_strong_1, D2_out_map_strong_1 = model_D2(pred_cat_strong_1)
            D2_out_strong_2, D2_out_map_strong_2 = model_D2(pred_cat_strong_2)

            pred_sel_strong_1,pred_sel_strong_2, labels_sel, mixed_img_strong_1, mixed_img_strong_2, mixed_label, good_count, bad_count = find_good_bad_maps(D1_out_weak, F.softmax(outputs_weak, dim=1), outputs_strong_1, outputs_strong_2, weak_inputs, strong_inputs, label_batch[:args.labeled_bs], args)

            if good_count > 0:
                labels_sel = Variable(labels_sel.long()).cuda(non_blocking=True)
                loss_st_ce_strong_1 = F.cross_entropy(pred_sel_strong_1, labels_sel, reduction='mean')
                loss_st_ce_strong_2 = F.cross_entropy(pred_sel_strong_2, labels_sel, reduction='mean')
                loss_st_dice_strong_1 = Dice_Funtion(pred_sel_strong_1, labels_sel.unsqueeze(dim=1))
                loss_st_dice_strong_2 = Dice_Funtion(pred_sel_strong_2, labels_sel.unsqueeze(dim=1))
                loss_st_good_strong_1 = 0.5 * (loss_st_ce_strong_1 + loss_st_dice_strong_1)
                loss_st_good_strong_2 = 0.5 * (loss_st_ce_strong_2 + loss_st_dice_strong_2)
                loss_st_good = (loss_st_good_strong_1+loss_st_good_strong_2)/2
            else:
                loss_st_good = 0.0

            if bad_count > 0:
                mixed_label = Variable(mixed_label.long()).cuda(non_blocking=True)
                mixed_pred_strong, _ = model(torch.cat((mixed_img_strong_1,mixed_img_strong_2)))
                mixed_pred_strong_1, mixed_pred_strong_2 = mixed_pred_strong.chunk(2)
                loss_st_ce_strong_1 = F.cross_entropy(mixed_pred_strong_1, mixed_label, reduction='mean')
                loss_st_ce_strong_2 = F.cross_entropy(mixed_pred_strong_2, mixed_label, reduction='mean')
                loss_st_dice_strong_1 = Dice_Funtion(mixed_pred_strong_1, mixed_label.unsqueeze(dim=1))
                loss_st_dice_strong_2 = Dice_Funtion(mixed_pred_strong_2, mixed_label.unsqueeze(dim=1))
                loss_st_bad_strong_1 = 0.5 * (loss_st_ce_strong_1 + loss_st_dice_strong_1)
                loss_st_bad_strong_2 = 0.5 * (loss_st_ce_strong_2 + loss_st_dice_strong_2)
                loss_st_bad = (loss_st_bad_strong_1 + loss_st_bad_strong_2) / 2
            else:
                loss_st_bad = 0.0

            loss_st = loss_st_good + loss_st_bad

            D_gt_v = Variable(one_hot(label_batch[:args.labeled_bs].unsqueeze(dim=1), 2)).cuda(non_blocking=True)
            image_gt_netD = (volume_batch[:args.labeled_bs] - torch.min(volume_batch[:args.labeled_bs])) / (torch.max(volume_batch[:args.labeled_bs]) - torch.min(volume_batch[:args.labeled_bs]))

            D_gt_v_cat = torch.cat((D_gt_v, image_gt_netD), dim=1)
            D1_out_gt, D1_out_gt_map = model_D(D_gt_v_cat)

            loss_fm_weak = 0
            for yy in range(len(D1_out_map_weak)):
                loss_fm_weak += torch.mean(torch.abs(D1_out_gt_map[yy] - D1_out_map_weak[yy]))
            loss_fm_weak = loss_fm_weak / len(D1_out_gt_map)

            loss_fm_strong_1 = 0
            loss_fm_strong_2 = 0
            for yy in range(len(D2_out_map_weak)):
                loss_fm_strong_1 += torch.mean(torch.abs(D2_out_map_strong_1[yy] - D2_out_map_weak[yy]))
                loss_fm_strong_2 += torch.mean(torch.abs(D2_out_map_strong_2[yy] - D2_out_map_weak[yy]))

            loss_fm_strong_1 = loss_fm_strong_1/len(D2_out_map_weak)
            loss_fm_strong_2 = loss_fm_strong_2/len(D2_out_map_weak)
            loss_fm_strong = (loss_fm_strong_1+loss_fm_strong_2)/2


            loss_fm = loss_fm_weak+loss_fm_strong

            loss = supervised_loss  + args.lambda_fm * loss_fm + args.lambda_st * loss_st

            iter_num += 1
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/loss_fm', args.lambda_fm * loss_fm, iter_num)
            writer.add_scalar('train/loss_fm_weak', args.lambda_fm * loss_fm_weak, iter_num)
            writer.add_scalar('train/loss_fm_strong', args.lambda_fm * loss_fm_strong, iter_num)
            writer.add_scalar('train/loss_fm_strong_1', args.lambda_fm * loss_fm_strong_1, iter_num)
            writer.add_scalar('train/loss_fm_strong_2', args.lambda_fm * loss_fm_strong_2, iter_num)
            writer.add_scalar('train/loss_st', args.lambda_st * loss_st, iter_num)
            writer.add_scalar('train/loss_st_good', args.lambda_st * loss_st_good, iter_num)
            writer.add_scalar('train/loss_st_bad', args.lambda_st * loss_st_bad, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_seg: %03f, loss_seg_dice: %03f'%(iter_num, loss, loss_seg, loss_seg_dice))


            # train D
            model.eval()
            model_D.train()
            model_D2.train()
            for param in model_D.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
            for param in model.parameters():
                param.requires_grad = False
            outputs, _ = model(weak_inputs)
            outputs_strong, _ = model(strong_inputs)
            outputs_label, outputs_weak = outputs.chunk(2)
            outputs_strong_1, outputs_strong_2=outputs_strong.chunk(2)

            pred_cat_weak = torch.cat((F.softmax(outputs_weak, dim=1), weak_netD), dim=1).detach()
            pred_cat_strong_1 = torch.cat((F.softmax(outputs_strong_1, dim=1), strong_netD_s1), dim=1).detach()
            pred_cat_strong_2 = torch.cat((F.softmax(outputs_strong_2, dim=1), strong_netD_s2), dim=1).detach()


            bce_loss = nn.BCELoss()
            D_out_z, _ = model_D(pred_cat_weak)
            y_fake_ = Variable(torch.zeros(D_out_z.size(0), 1).cuda(non_blocking=True))
            loss_D_fake = bce_loss(D_out_z, y_fake_)

            # train with gt
            D_out_z_gt, _ = model_D(D_gt_v_cat)
            y_real_ = Variable(torch.ones(D_out_z_gt.size(0), 1).cuda(non_blocking=True))
            loss_D_real = bce_loss(D_out_z_gt, y_real_)

            loss_D = (loss_D_fake + loss_D_real) / 2.0
            Dopt.zero_grad()
            loss_D.backward()

            D_out_weak, _ = model_D2(pred_cat_weak)
            y_real_ = Variable(torch.ones(D_out_weak.size(0), 1).cuda(non_blocking=True))
            loss_D_fake_2 = bce_loss(D_out_weak, y_real_)

            # train with gt
            D_out_fake, _ = model_D2(torch.cat((pred_cat_strong_1,pred_cat_strong_2)))
            D_out_strong_1, D_out_strong_2 = D_out_fake.chunk(2)
            y_fake_ = Variable(torch.zeros(D_out_strong_1.size(0), 1).cuda(non_blocking=True))
            loss_D_real_2_strong_1 = bce_loss(D_out_strong_1, y_fake_)
            loss_D_real_2_strong_2 = bce_loss(D_out_strong_2, y_fake_)


            loss_D2 = (loss_D_fake_2 + loss_D_real_2_strong_1 + loss_D_real_2_strong_2) / 3.0
            Dopt_2.zero_grad()
            loss_D2.backward()

            Dopt.step()
            Dopt_2.step()

            writer.add_scalar('train/loss_D1', loss_D, iter_num)
            writer.add_scalar('train/loss_D2', loss_D2, iter_num)


            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_Pancreas(model, num_classes=args.num_classes, patch_size=args.patch_size, stride_xy=16, stride_z=4, nms=0)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()


            if iter_num >= args.self_max_iteration:
                break

        if iter_num >= args.self_max_iteration:
            iterator.close()
            break
    writer.close()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)

        # torch seed init.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        cudnn.benchmark = False
        cudnn.deterministic = True
    self_snapshot_path = "./model/Pancreas_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting AstMatch training.")
    if not os.path.exists(self_snapshot_path):
        os.makedirs(self_snapshot_path)
    if os.path.exists(self_snapshot_path + '/code'):
        shutil.rmtree(self_snapshot_path + '/code')
    shutil.copy('../code/Pancreas_AstMatch_train.py', self_snapshot_path)

    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, self_snapshot_path)


