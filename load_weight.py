import os.path as osp
import torch
import numpy as np
from utils.options import parse_options
from models_blurhandnet.selection_model import SelectionModel
from models_blurhandnet.generation_model import BaseModel
from torch.utils.data import DataLoader

opt, args = parse_options('options/train/baseline.yml')
# opt, args = parse_options('options/train/BlurHandNet_BH.yml')
sel_model = SelectionModel(opt).cuda()
gen_model = BaseModel(opt).cuda()

path_select = '/home/zzq/Yuming/BlurHand_RELEASE/experiments_old/k16t1_t4_jrc_nomsf/training_states/epoch_30.pth.tar'
weight_dict = torch.load(path_select, weights_only=True)
dddd = dict()
for k,v in weight_dict['network'].items():
    print('{}:\t{}'.format(k[7:], v.shape))
    if 'multihead_attn' in k:
        k = k.replace('multihead_attn', 'cross_attn')
    dddd[k[7:]] = v
load_info = sel_model.load_state_dict(dddd, strict=False)
print(load_info)

path_generate = '/home/zzq/Yuming/BlurHand_RELEASE/experiments/baseline_k16t4_sim/training_states/epoch_30.pth.tar'
weight_dict = torch.load(path_generate, weights_only=True)
dddd = dict()
for k,v in weight_dict['network'].items():
    print('{}:\t{}'.format(k[7:], v.shape))
    if 'multihead_attn' in k:
        k = k.replace('multihead_attn', 'cross_attn')
    dddd[k[7+10:]] = v
load_info = gen_model.load_state_dict(dddd, strict=False)
print(load_info)

new_weight = dict()
for k, v in gen_model.state_dict().items():
    new_weight['module.generator.'+k] = v
for k, v in sel_model.state_dict().items():
    new_weight['module.selector.'+k] = v

torch.save({'network': new_weight},'final_weight.pth.tar')