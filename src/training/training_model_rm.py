###
# model for BH, Supervised Learning with GT labels
# including making preditions and computing losses from inputs
###
import os.path as osp
from glob import glob
import torch
import torch.nn as nn

class TrainingModel(torch.nn.Module):
    # wrap all models together
    def __init__(self, opt, training=True):
        super().__init__()

        gen_name = opt['model']['generator']
        rew_name = opt['model']['selector']
        crit_name = opt['model']['criterion']

        self.generator = get_gen_model(gen_name, opt, training)
        self.selector = get_rew_model(rew_name, opt, training)
        self.criterion = get_criterion(crit_name, opt)
    
        load_model(self.generator, self.selector, opt)
        for p in self.generator.parameters():
            p.requires_grad = False
        self.mode_list = ['train', 'test']

    def forward(self, inputs, targets, meta_info, mode = None):
        if mode == 'train':
            with torch.no_grad():
                preds = self.generator(inputs, targets, meta_info, 'train')
                inputs.update(preds)
            preds_rm = self.selector(inputs, targets, meta_info, 'train')
            preds.update(preds_rm)
            loss = self.criterion(preds, targets, meta_info, 'train')
            return loss
        elif mode == 'test':
            with torch.no_grad():
                preds = self.generator(inputs, targets, meta_info, 'test')
                inputs.update(preds)
            preds_rm = self.selector(inputs, targets, meta_info, 'test')
            preds.update(preds_rm)
            res = self.criterion(preds, targets, meta_info, 'test')
            return res
        else:
            raise KeyError(f"arg `mode` should be one of {self.mode_list} but got {mode} !")
        
def get_gen_model(name, opt, training):
    if name == 'models.generation_model':
        from models.generation_model import BaseModel
        return BaseModel(opt, training)
    raise KeyError

def get_rew_model(name, opt, training):
    if name == 'models.selection_model':
        from models.selection_model import SelectionModel
        return SelectionModel(opt, training)
    raise KeyError

def get_criterion(name, opt):
    if name == 'criterion':
        from training.criterion import Criterion
        return Criterion(opt)
    if name == 'criterion_rm':
        from training.criterion_rm import Criterion
        return Criterion(opt)

    raise KeyError

def load_model(gen_model: torch.nn.Module, sel_model: torch.nn.Module, opt):
    exp_name = opt['name']
    exp_dir = osp.join('experiments', exp_name) # drop 'postfix'
    states_list = glob(osp.join(exp_dir, 'training_states', '*.pth.tar'))
    cur_epoch = max([int(file_name[file_name.find('epoch_') + 6:file_name.find('.pth.tar')])
                         for file_name in states_list])
    fpath = osp.join(exp_dir, 'training_states', 'epoch_{:02d}'.format(cur_epoch) + '.pth.tar')
    data = torch.load(fpath, weights_only=True)['network']
    weights = dict()
    for k, v in data.items():
        weights[k[17:]] = v  # drop 'module.generator.'
    info_g = gen_model.load_state_dict(weights, strict=False)
    info_s = sel_model.load_state_dict(weights, strict=False)
    return info_g, info_s