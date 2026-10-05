#!/usr/bin/python2.7

import torch
from runner import Runner
import os
import argparse
import random
import numpy as np

# seeds are freezed
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='debug', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--config', type=str)
    parser.add_argument('--no_test_selection', action='store_true',
                        help='strict protocol: no test selection; save last_checkpoint.pth')
    parser.add_argument('--exact_dir', action='store_true',
                        help='use the explicit experiment directory without a timestamp')
    parser.add_argument('--save_scores', type=str, default=None,
                        help='optional directory for continuous ED surrogate npz files')
    args = parser.parse_args()
    run = Runner(args)
    if args.eval:
        run.evaluate()
    else:
        run.train()


if __name__ == '__main__':
    main()
