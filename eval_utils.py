import torch 
import torch.nn as nn
from progress.bar import Bar

@torch.no_grad()
def validate(model, testloader, logger, configs, verbose=True):
    
    if configs.task == 'classification':
        total, correct = 0, 0
        bar = Bar('Testing', max=len(testloader))
        model.eval()
        top1_error = AverageMeter()
        top5_error = AverageMeter()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = torch.ones(1)
                single_top1_error, loss, single_top5_error = compute_singlecrop(
                    outputs=outputs, labels=targets, loss=loss, 
                    top5_flag=True, mean_flag=True)
                top1_error.update(single_top1_error)
                top5_error.update(single_top5_error)
                #_, predicted = outputs.max(1)
                #total += targets.size(0)
                #correct += predicted.eq(targets).sum().item()
                #acc = correct / total

                bar.suffix = f'({batch_idx + 1}/{len(testloader)}) | ETA: {bar.eta_td} | top1: {100.0 - top1_error.avg} | top5: {100.0 - top5_error.avg}'
                bar.next()
        if verbose:
            logger.info('\nFinal result: top1 %.3f%% , top5 %.3f%%' % (100.0 - top1_error.avg, 100.0 - top5_error.avg))
        bar.finish()
        model.train()
        return 100.0 - top1_error.avg, 100.0 - top5_error.avg

    elif configs.task == 'detection':
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                logger.info(f'{inputs.shape}, {targets}')
                c = input()
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                c = input()

def compute_singlecrop(outputs, labels, loss, top5_flag=False, mean_flag=False):
    with torch.no_grad():
        if isinstance(outputs, list):
            top1_loss = []
            top1_error = []
            top5_error = []
            for i in range(len(outputs)):
                top1_accuracy, top5_accuracy = accuracy(outputs[i], labels, topk=(1, 5))
                top1_error.append(100 - top1_accuracy)
                top5_error.append(100 - top5_accuracy)
                top1_loss.append(loss[i].item())
        else:
            top1_accuracy, top5_accuracy = accuracy(outputs, labels, topk=(1,5))
            top1_error = 100 - top1_accuracy
            top5_error = 100 - top5_accuracy
            top1_loss = loss.item()

        if top5_flag:
            return top1_error, top1_loss, top5_error
        else:
            return top1_error, top1_loss

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        reset all parameters
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        update parameters
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count