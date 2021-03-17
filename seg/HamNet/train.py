import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import SGD
from torch.utils.data import DataLoader

import settings
from dataset import TrainDataset, ValDataset
from metric import cal_scores, fast_hist
from network import HamNet
from sync_bn.nn.modules import patch_replication_callback

logger = settings.logger


def params_count(model, print_shape=False):
    params = 0
    for module in model.parameters():
        params += np.prod(module.shape)
        if print_shape:
            print(module.shape)

    print('Total params: %f M' % (params / 1024 / 1024))


def get_params(model, key):
    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                yield m[1].weight
    if key == '1y':
        for m in model.named_modules():
            if isinstance(m[1], _BatchNorm):
                if m[1].weight is not None:
                    yield m[1].weight
        for m in model.named_parameters():
            if 'coef' in m[0]:
                yield m[1]
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], _BatchNorm):
                if m[1].bias is not None:
                    yield m[1].bias


def ensure_dir(dir_path):
    if not osp.isdir(dir_path):
        os.makedirs(dir_path)


def poly_lr_scheduler(opt, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    opt.param_groups[0]['lr'] = 1 * new_lr
    opt.param_groups[1]['lr'] = 1 * new_lr
    opt.param_groups[2]['lr'] = 2 * new_lr


class Session(object):
    def __init__(self, split='trainaug', dtype=None,
                 val=True):
        torch.manual_seed(66)
        torch.cuda.manual_seed_all(66)

        self.log_dir = osp.join(settings.LOG_DIR, settings.EXP_NAME)
        self.model_dir = osp.join(settings.MODEL_DIR, settings.EXP_NAME)

        ensure_dir(self.log_dir)
        ensure_dir(self.model_dir)

        logger.info('set log dir as %s' % self.log_dir)
        logger.info('set model dir as %s' % self.model_dir)

        self.step = 1
        self.best_mIoU = 0
        self.writer = SummaryWriter(self.log_dir)
        
        self.split = split

        train_set = TrainDataset(split=split)
        self.train_loader = DataLoader(
            train_set, batch_size=settings.TRAIN_BATCH_SIZE, pin_memory=True,
            num_workers=settings.NUM_WORKERS, shuffle=True, drop_last=True)

        val_set = ValDataset(split='val')
        self.val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                                     num_workers=settings.NUM_WORKERS,
                                     drop_last=False)

        self.net = HamNet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        params_count(self.net)

        self.opt = SGD(
            params=[
                {
                    'params': get_params(self.net, key='1x'),
                    'lr': 1 * settings.LR,
                    'weight_decay': settings.WEIGHT_DECAY,
                },
                {
                    'params': get_params(self.net, key='1y'),
                    'lr': 1 * settings.LR,
                    'weight_decay': 0,
                },
                {
                    'params': get_params(self.net, key='2x'),
                    'lr': 2 * settings.LR,
                    'weight_decay': 0.0,
                }],
            momentum=settings.LR_MOM)

        self.net = DataParallel(self.net)
        patch_replication_callback(self.net)

    def write(self, out):
        for k, v in out.items():
            self.writer.add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            '{}: {:.4g}'.format(k, v)
            for k, v in out.items()]
        logger.info(' '.join(outputs))

    def save_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        obj = {
            'net': self.net.module.state_dict(),
            'step': self.step,
            'best_mIoU': self.best_mIoU
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name, reset_steps=False):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path,
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.error('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

        if not reset_steps:
            self.step = obj['step'] + 1
        else:
            self.step = 1

        if 'best_mIoU' in obj:
            self.best_mIoU = obj['best_mIoU']

    def train_batch(self, image, label):
        loss = self.net(image, label)

        # Use the code below when setting HAM to CD.
        # loss, now = self.net(image, label)
        # delta = self.net.module.hamburger.online_update(now)
        # self.writer.add_scalar('bases_delta', delta, self.step)

        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def infer_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()
        logit = self.net(image)

        pred = logit.max(dim=1)[1]

        return fast_hist(label, pred)

    @torch.no_grad()
    def val(self, prefix='', logging=False, steps=None):
        self.net.eval()

        hist = 0
        for image, label in self.val_loader:
            hist += self.infer_batch(image, label)

        scores, cls_iu = cal_scores(hist.cpu().numpy())

        mIoU = scores['mIoU']
        if mIoU > self.best_mIoU:
            self.best_mIoU = mIoU

            if self.split == 'trainval':
                path = 'best.pth'
            else:
                path = 'best_trainaug.pth'

            self.save_checkpoints(path)

        steps = self.step if steps is None else steps
        for k, v in scores.items():
            self.writer.add_scalar(prefix+k, v, steps)

        if logging:
            logger.info('')
            logger.info('Total Scores')

            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))

            logger.info('')
            logger.info('Class Scores')
            for k, v in cls_iu.items():
                logger.info('%s-%f' % (k, v))
        else:
            self.net.train()


def train_main(ckp_name='latest.pth',
               split='trainaug', dtype=None,
               val=True, reset_steps=False):
    sess = Session(split=split, dtype=dtype)
    sess.load_checkpoints(ckp_name, reset_steps)

    loader = iter(sess.train_loader)
    sess.net.train()

    while sess.step <= settings.ITER_MAX:
        poly_lr_scheduler(
            opt=sess.opt,
            init_lr=settings.LR,
            iter=sess.step,
            lr_decay_iter=settings.LR_DECAY,
            max_iter=settings.ITER_MAX,
            power=settings.POLY_POWER)

        try:
            image, label = next(loader)
        except StopIteration:
            loader = iter(sess.train_loader)
            image, label = next(loader)

        loss = sess.train_batch(image, label)
        out = {'loss': loss}
        sess.write(out)

        if sess.step % settings.ITER_SAVE == 0:
            save_name = 'latest.pth'
            sess.save_checkpoints(save_name)

        if val and sess.step % settings.ITER_VAL == 0:
            sess.val()

        sess.step += 1
    
    if split == 'trainval':
        save_name = 'final.pth'
    else:
        save_name = 'final_trainaug.pth'

    sess.save_checkpoints(save_name)

    if val:
        sess.val(logging=True)
