"""
In this file, we aim to train the fcos network.
The parameters of the fcos network is fixed to use the slim vovnet

"""

import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch.nn as nn

import sys

sys.path.append('/nethome/jbang36/k_amoeba')

### we will make modification to this
from fixmatch.dataset import DATASET_GETTERS
from fixmatch.utils import AverageMeter, accuracy

from fixmatch.models.ema import ModelEMA

from fixmatch.models.ssd.argument import get_args

from fixmatch.models.ssd.ssd import SSD300
from fixmatch.dataset.bdd import collate_fn as bdd_collate_fn
from fixmatch.dataset.bdd import collate_fn2 as bdd_unlabeled_collate_fn

from fixmatch.distributed import get_rank, all_gather
from fixmatch.evaluate_coco import evaluate_fixmatch, evaluate

from pprint import PrettyPrinter
#from benchmark.ssd_sgr.eval import evaluate_fixmatch as evaluate_voc
from fixmatch.utils.ssd_sgr_utils import label_map


logger = logging.getLogger(__name__)
best_acc = 0


def main():
    args = get_fixmatch_arguments()
    global best_acc

    ###########################################################################
    ################### ----------------------------------- ###################
    ###########################################################################

    ### 0. random parameters

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}", )

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    ###########################################################################
    ################### ----------------------------------- ###################
    ###########################################################################



    ###########################################################################
    ################### ----------------------------------- ###################
    ###########################################################################

    ### 2. load the dataset -- we will use bdd
    args.path = 'daytime'
    args.path2 = 'night'

    data_dir = '/srv/data/jbang36'

    if args.dataset == 'pascal_voc':
        from fixmatch.utils.ssd_sgr_utils import label_map

        labeled_dataset, unlabeled_dataset, test_day_dataset = DATASET_GETTERS[args.dataset](
            args, data_dir)
        test_night_dataset = None
        n_classes = len(label_map)  # number of different types of objects
        args.num_classes = n_classes
        lr = 1e-3  # learning rate
        momentum = 0.9  # momentum
        weight_decay = 5e-4  # weight decay

    elif args.dataset == 'pascal_bdd_day_night':
        from fixmatch.utils.bdd_utils import label_map
        labeled_dataset, unlabeled_dataset, (test_day_dataset, test_night_dataset) = DATASET_GETTERS[args.dataset](
            args, data_dir)
        n_classes = len(label_map)  # number of different types of objects
        args.num_classes = n_classes
        lr = 1e-3  # learning rate
        momentum = 0.9  # momentum
        weight_decay = 5e-4  # weight decay


    elif args.dataset == 'pascal_bdd':
        from fixmatch.utils.bdd_utils import label_map

        labeled_dataset, unlabeled_dataset, test_day_dataset = DATASET_GETTERS[args.dataset](
            args, data_dir)
        test_night_dataset = None
        n_classes = len(label_map)  # number of different types of objects
        args.num_classes = n_classes
        lr = 1e-3  # learning rate
        momentum = 0.9  # momentum
        weight_decay = 5e-4  # weight decay


    else:
        labeled_dataset, unlabeled_dataset, (test_day_dataset, test_night_dataset) = DATASET_GETTERS[args.dataset](
        args, data_dir)
        args.num_classes = 11  ## number of classes in BDD is 11
        lr = 1e-5  # learning rate
        momentum = 0.5  # momentum
        weight_decay = 5e-4  # weight decay

    print(f"Loaded dataset: {args.dataset}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_width, args.image_height}")
    ###debugging
    labeled_im, _, _ = labeled_dataset[0]
    print(f"Labeled image size: {labeled_im.size()}")
    #(unlabeled_im, _), _, _ = unlabeled_dataset[0]
    #print(f"Unlabeled image size: {unlabeled_im.size()}")

    ### for fcos, we must get the arguments from the argument.py
    ### 1. create the model

    fcos_args = get_args(args)
    fcos_args.size_divisible = 0
    model = SSD300(args)

    print(f"loading model: SSD300...")
    print("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    model.to(args.device)


    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=bdd_collate_fn(fcos_args))

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=bdd_unlabeled_collate_fn(fcos_args))

    test_day_loader = DataLoader(
        test_day_dataset,
        sampler=SequentialSampler(test_day_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=bdd_collate_fn(fcos_args))

    if not (args.dataset == 'pascal_voc' or args.dataset == 'pascal_bdd'):
        test_night_loader = DataLoader(
            test_night_dataset,
            sampler=SequentialSampler(test_night_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=bdd_collate_fn(fcos_args))
    else:
        test_night_loader = None

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank == 0:
        torch.distributed.barrier()

    ###########################################################################
    ################### ----------------------------------- ###################
    ###########################################################################

    #### TODO: from this part and onwards, not sure if necessary

    biases = list()
    not_biases = list()


    #lr = args.lr
    #momentum = 0.9

    args.train_len = len(labeled_trainloader)

    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                lr=lr, momentum=momentum, weight_decay=weight_decay)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    train_ssd(args, labeled_trainloader, unlabeled_trainloader,
                   [test_day_loader, test_night_loader], [test_day_dataset, test_night_dataset],
                   model, optimizer, ema_model, scheduler)


def train_ssd(args, labeled_trainloader, unlabeled_trainloader, test_loaders, test_datasets,
                   model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_labeled_time = AverageMeter()
    forward_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    # mask_probs = AverageMeter() TODO: let's see if this makes any difference
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    model.to(args.device)
    decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
    decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate

    decay_lr_at = [it // (args.train_len // 32) for it in decay_lr_at]

    for epoch in range(args.start_epoch, args.epochs):

        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)


        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x, _ = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x, _ = labeled_iter.next()

            try:
                ### for the unlabeled iter, we get the weak augmented and strong augmented
                inputs_u_w, inputs_u_s, _ = unlabeled_iter.next()
                # (inputs_u_w, inputs_u_s), _, _ = unlabeled_iter.next()

            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                # (inputs_u_w, inputs_u_s), _, _ = unlabeled_iter.next()
                inputs_u_w, inputs_u_s, _ = unlabeled_iter.next()

            data_time.update(time.time() - end)

            optimizer.zero_grad()


            (loss1, _) = model(inputs_x.tensors, targets=targets_x, labeled=True)
            Lx = loss1

            forward_labeled_time.update(time.time() - end)

            ### (2) compute the loss between weakly unlabeled and strong unlabeled
            (loss2, _) = model(inputs_u_w.tensors,
                                    targets=inputs_u_s.tensors, labeled=False,
                                    threshold=args.threshold, image_sizes=inputs_u_w.sizes)

            if loss2 is None:  ### this occurs when there is no box proposal... maybe we need to see how many times this happens...

                Lu = torch.Tensor([0])
                loss = Lx

            else:
                loss2 = torch.nan_to_num(loss2)

                Lu = loss2
                loss = Lx + args.lambda_u * Lu

            forward_time.update(time.time() - end)


            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            # mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Forward Labeled: {frl: .3f}s Forward: {fr:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        frl=forward_labeled_time.avg,
                        fr=forward_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            #### instead of using the test loss... let's try doing the valid function
            assert (len(test_loaders) == 2)
            assert (len(test_datasets) == 2)
            test_day_loader, test_night_loader = test_loaders
            test_day_dataset, test_night_dataset = test_datasets
            print(f"DAY EVALUATION!!")
            day_results = valid(args, test_day_loader, test_day_dataset, test_model, args.device)
            if not (args.dataset == 'pascal_voc' or args.dataset == 'pascal_bdd'):
                print(f"NIGHT EVALUATION!!")
                night_results = valid(args, test_night_loader, test_night_dataset, test_model, args.device)

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)


            if args.dataset == 'pascal_bdd_day_night':
                args.writer.add_scalar('test/1.day_mAP', day_results, epoch)
                args.writer.add_scalar('test/2.night_mAP', night_results, epoch)

            elif not (args.dataset == 'pascal_voc' or args.dataset == 'pascal_bdd'):
                args.writer.add_scalar('test/1.day_mAP', day_results['bbox']['AP'], epoch)
                args.writer.add_scalar('test/2.day_mAP50', day_results['bbox']['AP50'], epoch)
                args.writer.add_scalar('test/3.day_mAP75', day_results['bbox']['AP75'], epoch)

                args.writer.add_scalar('test/4.night_mAP', night_results['bbox']['AP'], epoch)
                args.writer.add_scalar('test/5.night_mAP50', night_results['bbox']['AP50'], epoch)
                args.writer.add_scalar('test/6.night_mAP75', night_results['bbox']['AP75'], epoch)

            else:
                args.writer.add_scalar('test/1.day_mAP', day_results, epoch)

            # is_best = test_acc > best_acc
            # best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch'         : epoch + 1,
                'state_dict'    : model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'optimizer'     : optimizer.state_dict(),
                'scheduler'     : scheduler.state_dict(),
            }, True, args.out)  ### Currently, we just save all the models -- let's figure this out later

            """
            save_checkpoint({
                'epoch'         : epoch + 1,
                'state_dict'    : model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc'           : test_acc,
                'best_acc'      : best_acc,
                'optimizer'     : optimizer.state_dict(),
                'scheduler'     : scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

            """

    if args.local_rank in [-1, 0]:
        args.writer.close()


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    ### if checkpoint doesn't exist, we create it
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    epoch = state['epoch']
    if epoch % 20 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               f'model-epoch{epoch}.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


##########################################################################################
##########################################################################################
################################ EVALUATION FUNCTIONS ####################################
##########################################################################################
##########################################################################################
@torch.no_grad()
def valid_p1(loader, m, device):
    # if args.distributed:
    #    model = m.module

    torch.cuda.empty_cache()

    if isinstance(m, nn.DataParallel):
        model = m.module
    else:
        model = m

    model.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    preds = {}

    for images, targets, ids in pbar:
        model.zero_grad()

        images = images.to(device)

        #### pred is basically the BoxList object,,,
        pred = model(images.tensors, image_sizes=images.sizes,
                     train=False)  ### should the predictions be class predictions or the boxes????

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    return preds


@torch.no_grad()
def valid_p2(dataset, preds):
    results = evaluate(dataset, preds)
    del preds
    return results.results  ### should return an ordered dict


@torch.no_grad()
def valid(args, loader, dataset, m, device):
    if args.dataset in  ['pascal_voc', 'pascal_bdd', 'pascal_bdd_day_night']:
        pp = PrettyPrinter()
        from fixmatch.evaluate_pascal import evaluate_fixmatch as evaluate_voc
        APs, mAP = evaluate_voc(args, loader, m)
        pp.pprint(APs)

        return mAP


    else:

        preds = valid_p1(loader, m, device)
        results = valid_p2(dataset, preds)
        return results


def accumulate_predictions(predictions):
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions



def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))

##########################################################################################
##########################################################################################
######################### END OF EVALUATION FUNCTIONS ####################################
##########################################################################################
##########################################################################################


def get_fixmatch_arguments():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image-width', default=300, type=int)
    parser.add_argument('--image-height', default=400, type=int)
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--dataset', default='bdd_fcos', type=str,
                        choices=['cifar10', 'cifar100', 'bdd', 'bdd_fcos', 'pascal_voc', 'pascal_bdd', 'pascal_bdd_day_night'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=100,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='ssd', type=str,
                        choices=['wideresnet', 'resnext', 'fcos', 'ssd'],
                        help='architecture name')
    parser.add_argument('--total-steps', default=2 ** 20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='models/save/ssd/',
                        help='directory to output the result')  ### we changed this from results
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

    ### let's just first write the script to load the fcos and see if it works



