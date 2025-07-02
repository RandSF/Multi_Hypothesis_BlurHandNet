import numpy as np
import os
import os.path as osp
import torch
import warnings

from tqdm import tqdm
from utils.logger import ColorLogger
from utils.options import parse_options

from training.runner_bh import Runner

def main():
    # for visibility
    warnings.filterwarnings('ignore')
    
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    
    # load opt and args from yaml
    # opt, args = parse_options("options/test/pretrained_BlurHandNet_BH.yml")
    opt, args = parse_options("options/test/baseline.yml")

    dataset_name = opt['dataset']['name']
    eval_logger = ColorLogger(osp.join('experiments', opt['name'], 'results'), log_name=f'{dataset_name}_test_log.txt')
    eval_logger.info('Load checkpoint from {}_epoch_{}'.format(opt['name'], opt['test']['epoch']))
    tester = Runner(opt, args, None, training=False)
    
    eval_result = {}
    cur_sample_idx = 0
    eval_result = tester.evaluate()
   
    # for extra metric AUC:
    L = 100
    thresholds = torch.linspace(0, 50, L)
    jpe_middle = eval_result.get('jpe_current_sort', eval_result['jpe_current']).clone()    # [N, J], N is the total number of samples
    mask = ~torch.isnan(jpe_middle) # available samples
    jpe_middle[~mask] = torch.inf
    N = torch.sum(mask)
    # auc_res = torch.zeros(L)   # ratio of correct predictions
    # for i, th in enumerate(thresholds):
    #     auc_res[i] = torch.sum(jpe_middle<th) / N
    # # np.savetxt('ajpe.txt', jpe_middle.cpu().numpy())
    # # np.savetxt('auc.txt', auc_res.numpy(),)
    # eval_result['auc'] = torch.mean(auc_res[:-1]+auc_res[1:])/2
    for metric, value in eval_result.items():
        eval_logger.info(f"{metric}:\t{value.nanmean().item():.2f} mm")

if __name__ == '__main__':
    main()