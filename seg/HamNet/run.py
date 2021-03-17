import argparse
import os
import os.path as osp

import settings
from train import train_main
from eval import eval_main


def voc_test(eval_steps):
    train_main('latest.pth', split='trainaug')
    eval_main('best_trainaug.pth', eval_steps, 'best')
    eval_main('final_trainaug.pth', eval_steps, 'final')

    settings.LR = settings.LR / 10
    settings.ITER_MAX = settings.TEST_ITER_MAX
    settings.ITER_SAVE = settings.TEST_ITER_SAVE
    settings.ITER_VAL = settings.TEST_ITER_VAL
    train_main('final_trainaug.pth', split='trainval',
               val=True, reset_steps=True)
            
    eval_steps = [5, 6, 7]
    eval_main('best.pth', eval_steps, 'trainval_best')
    eval_main('final.pth', eval_steps, 'trainval_final')


def voc_val(eval_steps):
    train_main('latest.pth', split='trainaug')
    eval_main('best_trainaug.pth', eval_steps, 'best')
    eval_main('final_trainaug.pth', eval_steps, 'final')


if __name__ == '__main__':
    eval_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30]
        
    if settings.RUN_FOR_TEST:
        voc_test(eval_steps)
    else:
        voc_val(eval_steps)
