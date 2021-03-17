import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader

import settings
from dataset import ValDataset
from metric import cal_scores, fast_hist
from network import HamNet

logger = settings.logger


def ensure_dir(dir_path):
    if not osp.isdir(dir_path):
        os.makedirs(dir_path)


class EvalSession(object):
    def __init__(self, dt_split):
        self.model_dir = osp.join(settings.MODEL_DIR, settings.EXP_NAME)

        self.net = HamNet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.net = DataParallel(self.net)
        dataset = ValDataset(split=dt_split)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=2, drop_last=False)
        self.hist = 0

    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path,
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

    def inf_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1]
        self.hist += fast_hist(label, pred)


def eval_by_steps(ckp_name):
    sess = EvalSession('val')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()

    for _, [image, label] in enumerate(dt_iter):
        sess.inf_batch(image, label)

    logger.info('')
    logger.info('Total Scores')
    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))

    logger.info('')
    logger.info('Class Scores')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))

    return scores['mIoU']


def write_rec(mIoU_rec, suffix):
    eval_log_dir = osp.join(settings.EVAL_LOG_DIR, settings.EXP_NAME)
    ensure_dir(eval_log_dir)
    path = osp.join(eval_log_dir, f'{suffix}_rec')

    with open(path, 'w') as f:
        f.write(settings.EXP_NAME+'\n\n')
        f.write('model %s train_steps %d\n' % \
                (settings.HAM_TYPE, settings.TRAIN_STEPS))
        for k, v in mIoU_rec.items():
            f.write('eval_steps %d  mIoU %f\n' % (k, v))


def eval_main(ckp_name, eval_steps, suffix):
    mIoU_rec = dict()

    for steps in eval_steps:
        settings.EVAL_STEPS = steps
        logger.info('Current EVAL_STEPS: %d' % settings.EVAL_STEPS)
        mIoU = eval_by_steps(ckp_name)
        mIoU_rec[steps] = mIoU

    write_rec(mIoU_rec, suffix)


if __name__ == '__main__':
    eval_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]
    eval_main('best_trainaug.pth', eval_steps, 'best_retry')
    eval_main('final_trainaug.pth', eval_steps, 'final_retry')
