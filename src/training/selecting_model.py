###
# model for BH, Supervised Learning with GT labels
# including making preditions and computing losses from inputs
###
import torch
import torch.nn as nn

class SelectingModel(torch.nn.Module):
    def __init__(self, opt, training=True):
        super().__init__()

        pred_name = opt['model']['selector']
        crit_name = opt['model']['criterion']
        exec(f'from {pred_name} import SelectionModel')
        exec(f'from training.{crit_name} import Criterion_Selector')
        globals().update(locals())

        self.selector = SelectionModel(opt, training)
        self.criterion = Criterion_Selector(opt)
        
        self.mode_list = ['train-rm', 'test']

    def forward(self, preds, inputs, targets, meta_info, mode = None):
        if mode == 'train-rm':
            preds_rm = self.selector(preds, inputs, targets, meta_info, 'train')
            loss = self.criterion(preds, preds_rm, targets, meta_info, 'train')
            return loss
        
        elif mode == 'test':
            preds_rm = self.selector(preds, inputs, targets, meta_info, 'test')
            res = self.criterion(preds, preds_rm, targets, meta_info, 'test')
            return res
        
        else:
            raise KeyError(f"arg `mode` should be one of {self.mode_list} but got {mode} !")