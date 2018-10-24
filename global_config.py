import numpy as np
import torch
import logging


USE_CUDA = True
DELTA = 1e-4
BOS_ID = 0
EOS_ID = 1

logging.basicConfig(format='[%(name)s, %(levelno)2s]: %(message)s', level=logging.INFO)


# ANCHOR_LIST = [
#     # more global component
#     [1./4, 1./2], [2./4, 1./2], [3./4, 1./2],  # 2, w=1/2
#     [1./6, 1./3], [3./6, 1./3], [5./6, 1./3],  #[1./3, 2./3], [2./3, 2./3], # 3, w=1/3
#     #  localize component
#     [1./16, 1./8], [3./16, 1./8], [5./16, 1./8], [7./16, 1./8], [9./16, 1./8], [11./16, 1./8], [13./16, 1./8], [15./16, 1./8], # 4, w=1./8
#     [1./2, 1.0]
# ]


ANCHOR_LIST = [

    # more global component
    [1./4, 1./2], [2./4, 1./2], [3./4, 1./2],  # 2, w=1/2
    [1./6, 1./3], [3./6, 1./3], [5./6, 1./3],  #[1./3, 2./3], [2./3, 2./3], # 3, w=1/3
    #  localize component
    [1./16, 1./8], [3./16, 1./8], [5./16, 1./8], [7./16, 1./8], [9./16, 1./8], [11./16, 1./8], [13./16, 1./8], [15./16, 1./8], # 4, w=1./8
    [1./2, 1.0]
]
logger = logging.getLogger('setup')

# setup random seeds
torch.manual_seed(233)
np.random.seed(233)
logger.info('SET ALL RANDOM SEED %d', 233)

assert USE_CUDA, "Can not use cpu version for bugs in pytorch and python.multiprocessing"
# setup default torch
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    from_numpy = lambda ndarray: torch.from_numpy(ndarray).cuda()
    logger.warning('USE CUDA')
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    from_numpy = torch.from_numpy

