## ssh -L 16006:127.0.0.1:16006 mb2775@ogg.cs.bath.ac.uk
import torch
from models import XNOR_VGG
import torchvision
from torchvision import transforms
import argparse
import binop
import torch.utils.tensorboard  as tensorboard
import os
from datetime import datetime
import time

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # end = time.time()
    bin_op.binarization()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = input.to(device)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        t0 = time.time()
        output = model(input_var)
        batch_time.update(time.time() - t0)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    bin_op.restore()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./ImageNet', help='path to dataset, default = ./ImageNet')
parser.add_argument('--attention', action='store_true', help='use attention branch model')
parser.add_argument('--imgres', type=int, default=224, help='image input and output resolution, default = 352')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs,  default = 100')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate,  default = 0.01')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum,  default = 0.9')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay,  default = 0.0005')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size,  default = 10')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin, default = 0.5')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate, default = 0.1')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate,  default = 50')
args = parser.parse_args()

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

device = torch.device('cpu')
model = XNOR_VGG().to(device)
# model = torchvision.models.vgg16(pretrained=True).to(device)
print('Model loaded')


b_model = XNOR_VGG(state_dict=model.features.state_dict()).to(device)
save_path = 'ckpts/{}/'.format(model.name)
torch.save(b_model.state_dict(), '{}{}.pth'.format(save_path, 'bin'))

model.eval()
with torch.no_grad():
    n = 100
    input = torch.rand([n, 1, 3, args.imgres, args.imgres]).to(device)
    t0 = time.time()
    for i in input:
        pred = model(i)
    avg_t = (time.time() - t0) / n
print('Inference time', avg_t)
print('FPS', 1/avg_t)

b_model.eval()
with torch.no_grad():
    n = 100
    input = torch.rand([n, 1, 3, args.imgres, args.imgres]).to(device)
    t0 = time.time()
    for i in input:
        pred = b_model(i)
    avg_t = (time.time() - t0) / n
print('Inference time', avg_t)
print('FPS', 1/avg_t)

transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((args.imgres, args.imgres)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = torchvision.datasets.ImageNet(args.dataset, split='train', transform=transform)
dataset_val = torchvision.datasets.ImageNet(args.dataset, split='val', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
total_steps = len(loader)
print('Dataset loaded')

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
bin_op = binop.BinOp(model)

writer = tensorboard.SummaryWriter(os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S')))

criterion = torch.nn.CrossEntropyLoss().to(device)

for epoch in range(args.epoch):
    lr_lambda = lambda epoch: args.decay_rate ** (epoch // args.decay_epoch)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    validate(loader_val, model, criterion)

    for step, sample in enumerate(loader, start=1):
        global_step = epoch * total_steps + step
        input, target = sample
        target_var = torch.autograd.Variable(target).to(device)
        input = input.to(device)
        target = input.to(device)
        bin_op.binarization()
        output = model(input)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()

        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()

        writer.add_scalar('Loss/Total Loss', float(loss), global_step)

        if step % 100 == 0 or step == total_steps:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                  format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch+1, args.epoch, step, total_steps, loss.data))


        if epoch % 5 == 0:
            save_path = 'ckpts/{}/'.format(model.name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), '{}{}.pth.{:03d}'.format(save_path, model.name, epoch))
