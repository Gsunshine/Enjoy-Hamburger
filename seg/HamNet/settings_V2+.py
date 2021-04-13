import logging

import numpy as np
import torch

from data_settings import *

# Data settings
MEAN = torch.Tensor(np.array([0.485, 0.456, 0.406]))
STD = torch.Tensor(np.array([0.229, 0.224, 0.225]))
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
TEST_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
CROP_SIZE = 513
IGNORE_LABEL = 255
NUM_WORKERS = 64


# Training settings
RUN_FOR_TEST = True

TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE   = 1

ITER_MAX  = 60000
ITER_SAVE = 1000
ITER_VAL  = 1000

TEST_ITER_MAX = 20000
TEST_ITER_SAVE = 1000
TEST_ITER_VAL  = 1000

LR_DECAY = 10
LR = 9e-3
LR_MOM = 0.9
POLY_POWER = 0.9
WEIGHT_DECAY = 1e-4


# Network
N_CLASSES = 21
N_LAYERS = 101
STRIDE = 8
BN_MOM = 3e-4


# Hamburger
CHANNELS = 512
VERSION  = 'V2+'

HAM_TYPE = 'NMF'
DUAL = False
SPATIAL = True
RAND_INIT = True
ZERO_HAM = True

MD_S = 1
MD_D = 512
MD_R = 512

CHEESE_FACTOR = 1
TRAIN_STEPS = 6
EVAL_STEPS = 6

INV_T = 1
BETA = 0.1
ETA = 0.9


# Path
EXP_NAME = f'Ham{VERSION}_Res{N_LAYERS}_{HAM_TYPE}_C{CHANNELS}_S{MD_S}_D{MD_D}_R{MD_R}_steps{TRAIN_STEPS}'
LOG_DIR = './log_dir'
EVAL_LOG_DIR = './eval_log/'


# Tensorboard
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
