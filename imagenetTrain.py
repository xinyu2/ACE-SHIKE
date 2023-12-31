import argparse
import datetime
import os
# import shutil
# from tkinter import N
# from turtle import color
import torch
import torch.optim as optim
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast, GradScaler
from autoaugment import CIFAR10Policy, ImageNetPolicy, Cutout
from datasets.dataset_lt_data import LT_Dataset
from networks import resnet50
from utils import *

# data transform settings
normalize = transforms.Normalize(
    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
data_transforms = {
    'base_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    # augmentation adopted in balanced meta softmax & NCL
    'advanced_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize
    ]),
    # augmentation adopted in balanced meta softmax & NCL
    'advanced_imagenet_train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet-LT Training')
parser.add_argument('--outf', default='./outputs/', help='folder to output images and model checkpoints')
parser.add_argument('--pre_epoch', default=0, help='epoch for pre-training')
parser.add_argument('--epochs', default=200, help='epoch for augmented training')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--learning_rate', default=0.05)
parser.add_argument('--seed', default=123, help='keep all seeds fixed')
parser.add_argument('--re_train', default=True, help='implement cRT')
parser.add_argument('--cornerstone', default=180)
parser.add_argument('--num_classes', default=1000, type=int, help='number of classes ')
parser.add_argument('--num_exps', default=3, type=int, help='number of experts')
parser.add_argument('--lossfn', default='ori', type=str, help='loss-function (ori, ace)')
parser.add_argument('--L1', default=0.1, type=float, help='lambda1-of-ace1')
parser.add_argument('--L2', default=0.4, type=float, help='lambda2-of-ace1')
parser.add_argument('--L3', default=0.8, type=float, help='lambda3-of-ace1')
parser.add_argument('--f0', default=0.4, type=float, help='f0-of-ace1')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p',
                    '--print-freq',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
args = parser.parse_args()
lam1 = args.L1
lam2 = args.L2
lam3 = args.L3
f0 = args.f0
lambdas = [lam1, lam2, lam3]
parmstr = "-".join([str(l) for l in [lam1, lam2, lam3, f0]])
savename = 'ImageNet-LT-' + parmstr

def main():
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    # imbalance distribution
    # img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
    ncls = args.num_classes
    
    
    dataroot = './datasets/data'
    dir_train_txt = os.path.join(dataroot, 'data_txt', 'ImageNet_LT_train.txt')
    dir_test_txt = os.path.join(dataroot, 'data_txt', 'ImageNet_LT_test.txt')
    lt_root = os.path.join(dataroot, 'ImageNet')
    train_set = LT_Dataset(lt_root, dir_train_txt, data_transforms['advanced_imagenet_train'])
    test_set = LT_Dataset(lt_root, dir_test_txt, data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=16)

    cls_num_list = train_set.get_per_class_num()
    num = np.array(cls_num_list)
    args.label_dis = num
    mask = get_mask(cls_num_list, ncls, lambdas)
    # print(f"mask={mask[0], mask[50], mask[99]}, size of testset_data:{test_set.__len__()}")
    best_acc1 = .0

    model = resnet50(num_classes=ncls, use_norm=True,
                     num_exps=args.num_exps).to(device)

    # optimizers and schedulers for decoupled training
    optimizer_feat = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer_crt = optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler_feat = CosineAnnealingLRWarmup(
        optimizer=optimizer_feat,
        T_max=args.epochs - 20,
        eta_min=0.0,
        warmup_epochs=5,
        base_lr=args.learning_rate,
        warmup_lr=0.15
    )
    scheduler_crt = CosineAnnealingLRWarmup(
        optimizer=optimizer_crt,
        T_max=20,
        eta_min=0.0,
        warmup_epochs=5,
        base_lr=args.learning_rate,
        warmup_lr=0.1
    )

    criterion = nn.CrossEntropyLoss().cuda()

    if args.evaluate:
        validate(test_loader, model, criterion, 179, args)
        return

    # proceeding with torch apex
    scaler = GradScaler()

    for epoch in range(0, args.epochs):  # args.start_epoch

        # freezing shared parameters
        if epoch >= args.cornerstone:
            for name, param in model.named_parameters():
                if name[:14] != "rt_classifiers":  # DDP NAME module.classifiers
                    param.requires_grad = False

        # train for one epoch
        train(train_loader if epoch >= args.cornerstone else train_loader, model, scaler,
              optimizer_crt if epoch >= args.cornerstone else optimizer_feat, epoch, mask, f0, args)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion, epoch, args)

        # adjust learning rate
        if epoch >= args.cornerstone:
            scheduler_crt.step()
        else:
            scheduler_feat.step()

        # record best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'architecture': "resnet32",
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, feat=(epoch < args.cornerstone), epochs=args.epochs, folder=args.outf, filename=savename)
    print("Training Finished, TotalEPOCH=%d" % args.epochs)

def get_mask(cls_num_list, ncls, lambdas):
    mask = torch.eye(ncls, dtype=torch.float32, requires_grad=False).cuda()
    for k in range(ncls):
        if cls_num_list[k]>=100:
            mask[k,k] = lambdas[0]
        elif cls_num_list[k]>=20:
            mask[k,k] = lambdas[1]
        else:
            mask[k,k] = lambdas[2]
    return mask

def mix_outputs(outputs, labels, balance=False, label_dis=None, lossfn='ori', mask=None, f0=None):
    logits_rank = outputs[0].unsqueeze(1)
    for i in range(len(outputs) - 1):
        logits_rank = torch.cat(
            (logits_rank, outputs[i+1].unsqueeze(1)), dim=1)
    max_tea, max_idx = torch.max(logits_rank, dim=1) # max-logit of 3 experts
    # min_tea, min_idx = torch.min(logits_rank, dim=1)
    non_target_labels = torch.ones_like(labels) - labels

    avg_logits = torch.sum(logits_rank, dim=1) / len(outputs)
    # print(f"outputs={outputs[0].shape}/{len(outputs)} \t labels={labels.shape} \t avg-logits={avg_logits.shape} \t non-target-labels={non_target_labels.shape}")
    non_target_logits = (-30 * labels) + avg_logits * non_target_labels # -30 is just some big negative value

    _hardest_nt, hn_idx = torch.max(non_target_logits, dim=1)

    hardest_idx = torch.zeros_like(labels)
    hardest_idx.scatter_(1, hn_idx.data.view(-1, 1), 1) # one-hot of hardest-negative-sample
    hardest_logit = non_target_logits * hardest_idx # avg-logit of the hardest class
    
    rest_nt_logits = max_tea * (1 - hardest_idx) * (1 - labels) # max-logit for other-wrong-predictions
    reformed_nt = rest_nt_logits + hardest_logit # max-logit of other-wrong-predictions, avg-logit of hardest-negative-prediction

    preds = [F.softmax(logits) for logits in outputs] # softmax (probability) from 3 experts

    reformed_non_targets = []
    for i in range(len(preds)):
        target_preds = preds[i] * labels

        target_preds = torch.sum(target_preds, dim=-1, keepdim=True)
        target_min = -30 * labels # -30 is just some big negative value
        target_excluded_preds = F.softmax(
            outputs[i] * (1 - labels) + target_min) # softmax (probability) of the wrong-predictions
        reformed_non_targets.append(target_excluded_preds)

    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1) # expand to the shape of labels
    # print(f"label-dis={label_dis}, shape={label_dis.shape} ")
    loss = 0.0
    if balance == True:
        # lossfn = 'ori' # disable ace1 after epoch 180
        for i in range(len(outputs)):
            if lossfn == 'ori':
                tmp = soft_entropy(outputs[i] + label_dis.log(), labels)
            else:
                tmp = ace1(outputs[i] + label_dis.log(), labels, mask=mask, f0=f0)
            loss += tmp
    else:
        for i in range(len(outputs)):
            # base ce
            if lossfn == 'ori':
                tmp = soft_entropy(outputs[i], labels)
            else:
                tmp = ace1(outputs[i], labels, mask=mask, f0=f0)
            loss += tmp        
            # hardest negative suppression
            loss += 10.0 * \
                F.kl_div(
                    torch.log(reformed_non_targets[i]), F.softmax(reformed_nt))
            # mutual distillation loss
            for j in range(len(outputs)):
                if i != j:
                    loss += F.kl_div(F.log_softmax(outputs[i]),
                                     F.softmax(outputs[j]))

    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output


def train(train_loader, model, scaler, optimizer, epoch, mask, f0, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # worst_case_per_round = None
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = F.one_hot(target, num_classes=args.num_classes)
        # if i == 0:
        #     print(f"target={target.shape}, {target}")
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        # compute output
        with autocast():
            outputs = model(images, (epoch >= args.cornerstone))
            loss, output = mix_outputs(outputs=outputs, labels=target, balance=(
                epoch >= args.cornerstone), label_dis=args.label_dis, lossfn=args.lossfn, mask=mask, f0=f0)
        _, target = torch.max(target.data, 1)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            outputs = model(images, (epoch >= args.cornerstone))
            output = sum(outputs) / len(outputs)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
    return top1.avg


def save_checkpoint(state, is_best, feat, epochs, folder=None, filename=None):
    stage1_checkpoint = os.path.join(folder,f'{filename}_stage1.pth.tar')
    stage2_checkpoint = os.path.join(folder,f'{filename}_stage2.pth.tar')
    if is_best and feat:
        torch.save(state, stage1_checkpoint)
    elif is_best:
        torch.save(state, stage2_checkpoint)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    clock_start = datetime.datetime.now()
    main()
    clock_end = datetime.datetime.now()
    print(clock_end - clock_start)
