import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from model.data import CIFData
from model.data import collate_pool, get_train_val_test_loader
from model.model import TRANSFORMER


parser = argparse.ArgumentParser(
    description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                         help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                         help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                        help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--disable-save-torch', action='store_true',
                    help='Do not save CIF PyTorch data as .pkl files')
parser.add_argument('--clean-torch', action='store_true',
                    help='Clean CIF PyTorch data .pkl files')


args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()
# print(torch.cuda.is_available())
# print('args.cuda', args.cuda)

best_mae_error = 1e10

def main():
    global args, best_mae_error

    # load data
    dataset = CIFData(*args.data_options,
                      disable_save_torch=args.disable_save_torch)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    # make output folder if needed
    if not os.path.exists('output'):
        os.mkdir('output')

    # make and clean torch files if needed
    torch_data_path = os.path.join(args.data_options[0], 'cifdata')
    if args.clean_torch and os.path.exists(torch_data_path):
        shutil.rmtree(torch_data_path)
    if os.path.exists(torch_data_path):
        if not args.clean_torch:
            warnings.warn('Found cifdata folder at ' +
                          torch_data_path+'. Will read in .pkls as-available')
    else:
        os.mkdir(torch_data_path)

    # obtain target value normalizer
    if len(dataset) < 500:
        warnings.warn('Dataset has less than 500 data points. '
                        'Lower accuracy is expected. ')
        sample_data_list = [dataset[i] for i in range(len(dataset))]
    else:
        sample_data_list = [dataset[i] for i in
                            sample(range(len(dataset)), 500)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]    # dataset: (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
    orig_atom_fea_len = structures[0].shape[-1]  # structures:(atom_fea, nbr_fea, nbr_fea_idx)
    nbr_fea_len = structures[1].shape[-1]
    model = TRANSFORMER(args, orig_atom_fea_len, nbr_fea_len)

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    best_checkpoint = torch.load(os.path.join('output', 'model_best.pth.tar'))
    model.load_state_dict(best_checkpoint['state_dict'])
    print('---------Evaluate Best Model on Train Set---------------')
    validate(train_loader, model, criterion, normalizer, test=True,
             csv_name='train_results.csv')
    print('---------Evaluate Best Model on Val Set---------------')
    validate(val_loader, model, criterion, normalizer, test=True,
             csv_name='val_results.csv')
    print('---------Evaluate Best Model on Test Set---------------')
    validate(test_loader, model, criterion, normalizer, test=True,
             csv_name='test_results.csv')


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # input_:   input_[0]   atom features 
        #           input_[1]   neigbhor features 
        #           input_[2]   neighor number 
        #           input_[3]   atom Indices    
        if args.cuda:
            input_var = (Variable(input_[0].cuda(non_blocking=True)),
                         Variable(input_[1].cuda(non_blocking=True)),
                         input_[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input_[3]])
            # input: batch_atom_fea,batch_nbr_fea,batch_nbr_fea_idx,crystal_atom_idx
        else:
            input_var = input_

        # normalize target
        target_normed = normalizer.norm(target)
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                        epoch+1, i+1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mae_errors=mae_errors)
                    )


def validate(val_loader, model, criterion, normalizer, test=False, csv_name='test_results.csv'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input_, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input_[0].cuda(non_blocking=True)),
                             Variable(input_[1].cuda(non_blocking=True)),
                             input_[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input_[3]])
        else:
            with torch.no_grad():
                input_var = input_
        target_normed = normalizer.norm(target)
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if test:
            print_name = 'Test'
        else:
            print_name = 'Validation'

        if i % args.print_freq == 0:
            print(print_name+': [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                        i+1, len(val_loader), batch_time=batch_time, loss=losses,
                        mae_errors=mae_errors))

    if test:
        star_label = '**'
        with open(os.path.join('output', csv_name), 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                    mae_errors=mae_errors))
    return mae_errors.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename=os.path.join('output', 'checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('output', 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
