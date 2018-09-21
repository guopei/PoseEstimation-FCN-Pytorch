import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import transforms as transforms

from datetime import datetime
from utils import create_if_not_exists as cine
from utils import Tee
from utils import vis_square
import pickle
from fcn import *
from heatmap import *
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr_decay', default='50', type=int,
                    help='lr decay frequency')
parser.add_argument('--crop_size', default='256', type=int,
                    help='size of cropped image')
parser.add_argument('--visualize', dest='visualize', action='store_true',
                    help='visualize middle output')
parser.add_argument('--nparts', default='15', type=int,
                    help='number of keypoints')

best_prec1 = 0
time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

cine('logs')
Tee('logs/cmd_log_{}'.format(time_string), 'w')

unisize = 256
outsize = 64

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    global fig, ax1, ax2, ax3, ax4
    if args.visualize:
        plt.ion()
        plt.show()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    # create model
    model = fcn(pretrained=args.pretrained, nparts=args.nparts)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()     
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print(model)
    model = model.cuda()
    
    train_sampler = None
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_dataset = Heatmap(
        traindir,
        transforms.Compose([
            transforms.Resize((unisize, unisize)),
            #transforms.RandomCrop(unisize),
            transforms.RandomRotation(30),
            transforms.ResizeTarget((outsize, outsize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = Heatmap(
            valdir, transforms.Compose([
            transforms.Resize((unisize,unisize)),
            transforms.ResizeTarget((outsize, outsize)),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(train_loader, model, criterion, args.epochs-1)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pck = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, segmap, paths) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        segmap = segmap.cuda(async=True)
        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(segmap)

        # compute output
        outputs = model(input_var)
        loss = criterion(outputs, target_var)

        # measure pck and record loss
        pck_score = calc_pck(outputs.data, segmap)
        losses.update(loss.item(), inputs.size(0))
        pck.update(pck_score, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % args.print_freq == 0:

            if args.visualize:
                visualize(inputs.cpu().numpy()[0].transpose(1,2,0),
                        outputs.data.cpu().numpy()[0], 
                        target_var.data.cpu().numpy()[0])


            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss {loss.val:.3e} ({loss.avg:.3e}) '
                  'PCK {pck.val:.3f} ({pck.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, pck=pck))

       
def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    pck = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()


    with torch.no_grad():
        for i, (inputs, segmap, paths) in enumerate(val_loader):
            segmap = segmap.cuda(async=True)
            input_var = torch.autograd.Variable(inputs.cuda())
            target_var = torch.autograd.Variable(segmap)

            # compute output
            outputs = model(input_var)
            loss = criterion(outputs, segmap)

            # measure accuracy and record loss
            pck_score = calc_pck(outputs.data, segmap)
            losses.update(loss.item(), inputs.size(0))
            pck.update(pck_score, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i > 0 and i % args.print_freq == 0:

                if args.visualize:
                    visualize(inputs.cpu().numpy()[0].transpose(1,2,0),
                            outputs.data.cpu().numpy()[0], 
                            target_var.data.cpu().numpy()[0])

                print('Test: [{0}/{1}] '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'Loss {loss.val:.3e} ({loss.avg:.3e}) '
                      'PCK {pck.val:.3f} ({pck.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       pck=pck))

    print(' * PCK {pck.avg:.3f}'.format(pck = pck))
    return pck.avg

def save_checkpoint(state, is_best, filename='*_checkpoint.pth.tar'):
    work_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    save_dir = os.path.join('./checkpoints/', work_dir)
    cine(save_dir)

    filename = filename.replace('*', time_string)
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, time_string+'_model_best.pth.tar'))

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_decay))
    #optimizer.param_groups[0]['lr'] = lr
    #optimizer.param_groups[1]['lr'] = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])

def visualize(img, otpt, grth):

    pred = vis_square(otpt)
    gt = vis_square(grth)
    ax1.imshow(pred)
    ax2.imshow(gt)
    ax3.imshow(pred>0.5)
    img = (img - img.min()) / (img.max() - img.min())
    for j in range(grth.shape[0]):
        enlarge = cv2.resize(grth[j], (unisize, unisize))
        img += np.random.rand(3) * enlarge[...,np.newaxis].repeat(3,axis=2)
    img = (img - img.min()) / (img.max() - img.min())
    ax4.imshow(img)
    plt.draw()
    plt.pause(0.001)

def find_max_loc(heatmap):
    (batches, channels) = heatmap.shape[:2]
    locs = np.zeros((batches, channels, 2), np.uint32)
    for b in range(batches):
        for c in range(channels):
            locs[b,c] = np.unravel_index(heatmap[b,c].argmax(), heatmap[b,c].shape)
    
    return locs

def get_dists(preds, gts):
    (batches, channels) = preds.shape[:2]
    dists = np.zeros((channels, batches), np.int32)
    for b in range(batches):
        for c in range(channels):
            if gts[b, c, 0] > 0 and gts[b, c, 1] > 0:
                dists[c,b] = ((gts[b,c] - preds[b,c]) ** 2).sum() ** 0.5
            else:
                dists[c,b] = -1
    return dists

def within_threshold(dist, thr = 0.1):
    dist = dist[dist != -1]
    if len(dist) > 0:
        return (dist < thr * outsize).sum() / float(len(dist))
    else:
        return -1

def calc_pck(output, target):
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    preds = find_max_loc(output_np)
    gts = find_max_loc(target_np)
    dists = get_dists(preds, gts)
    acc = np.zeros(args.nparts, dtype=np.float32)
    avg_ccc = 0.0
    bad_idx_count = 0

    for i in range(args.nparts):
        acc[i] = within_threshold(dists[i])
        if acc[i] >= 0:
            avg_ccc = avg_ccc + acc[i]
        else:
            bad_idx_count = bad_idx_count + 1
  
    if bad_idx_count == args.nparts:
        return 0
    else:
        return avg_ccc / (args.nparts - bad_idx_count) * 100

def inverse_normalization(img, 
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]):
    
    for i, m, s in zip(img, mean, std):
        i *= s
        i += m
    return img


if __name__ == '__main__':
    main()
