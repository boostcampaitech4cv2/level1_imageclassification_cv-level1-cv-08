import os
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Trainer, Tester
from inference import Inference

from TRACER.config import getConfig
warnings.filterwarnings('ignore')
args = getConfig()


def main(args):
    print('<---- Training Params ---->')
    pprint.pprint(args)

    # Random Seed
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_path = os.path.join(args.model_path, args.dataset, f'TE{args.arch}_{str(args.exp_num)}')

    print('<----- Initializing inference mode ----->')
    Inference(args, save_path).test()


if __name__ == '__main__':
    main(args)