###
# model for BH, Supervised Learning with GT labels
# including making preditions and computing losses from inputs
###
import torch
import torch.nn as nn

class TrainingModel(torch.nn.Module):
    # wrap all models together
    def __init__(self, opt, training=True):
        super().__init__()

        gen_name = opt['model']['generator']
        crit_name = opt['model']['criterion']

        self.generator = get_gen_model(gen_name, opt, training)
        self.criterion = get_criterion(crit_name, opt)
        
        self.mode_list = ['train', 'train-rm', 'test']

    def forward(self, inputs, targets, meta_info, mode = None):
        if mode == 'train':
            preds = self.generator(inputs, targets, meta_info, 'train')
            loss = self.criterion(preds, targets, meta_info, 'train')
            return loss
        elif mode == 'test':
            preds = self.generator(inputs, targets, meta_info, 'test')
            preds.update(inputs)
            res = self.criterion(preds, targets, meta_info, 'test')
            return res
        else:
            raise KeyError(f"arg `mode` should be one of {self.mode_list} but got {mode} !")
        
def get_gen_model(name, opt, training):
    if name == 'models.generation_model':
        from models.generation_model import BaseModel
        return BaseModel(opt, training)
    raise KeyError

def get_criterion(name, opt):
    if name == 'criterion':
        from training.criterion import Criterion
        return Criterion(opt)
    raise KeyError