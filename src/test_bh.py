import numpy as np
import os
import os.path as osp
import torch
import warnings

from tqdm import tqdm
from utils.logger import ColorLogger
from utils.options import parse_options

from training import get_runner

def main():
    # for visibility
    warnings.filterwarnings('ignore')
    
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    
    # load opt and args from yaml
    opt, args = parse_options("options/test/baseline.yml")

    dataset_name = opt['dataset']['name']
    eval_logger = ColorLogger(osp.join('experiments', opt['name'], 'results'), log_name=f'{dataset_name}_test_log.txt')
    eval_logger.info('Load checkpoint from {}_epoch_{}'.format(opt['name'], opt['test']['epoch']))
    tester = get_runner(opt['model']['runner'], opt, args, eval_logger, training=False)
    
    eval_result = {}
    cur_sample_idx = 0
    eval_result = tester.evaluate()
   
    for metric, value in eval_result.items():
        eval_logger.info(f"{metric}:\t{value.nanmean().item():.2f} mm")

if __name__ == '__main__':
    main()