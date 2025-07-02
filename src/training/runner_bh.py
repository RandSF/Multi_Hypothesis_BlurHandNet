import math
import os
import os.path as osp
from glob import glob

import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DistributedSampler
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from training.lr_scheduler import CosineAnnealingWarmupRestarts

from data.multiple_dataset import MultipleDatasets

from tqdm import tqdm

from training import get_trainer

class Runner():
    def __init__(self, opt, args, logger, training):
        self.opt = opt
        self.exp_dir = osp.join('experiments', opt['name'])
        self.tb_dir = osp.join('tb_logger', opt['name'])
        self.cur_epoch = 0
        self.end_epoch = opt['train']['end_epoch'] if training else None
        self.world_size = dist.get_world_size() if training else None
        self.local_rank = args.local_rank
        self.device = torch.device("cuda", self.local_rank)
        self.logger = logger
        self.if_log_grad = opt.get('log_grad', False)


        if training:
            self._make_training_generator()
        self._make_evaluating_generator()
        
        if training:
            self._prepare_training()
        else:
            self._prepare_testing()

    def _make_training_generator(self):
        # dynamic dataset import   
        dataset_list = self.opt['dataset_list']
        for _, opt_data in dataset_list.items():
            dataset_name = opt_data['name']
            exec(f'from data.{dataset_name} import {dataset_name}')
        globals().update(locals())

        # data loader and construct batch generator
        trainset3d_loader = []
        trainset2d_loader = []
        
        dataset_list = self.opt['dataset_list']
        for _, opt_data in dataset_list.items():
            dataset_name = opt_data['name']
            # if self.local_rank == 0: self.logger.info(f"Creating dataset ... [{dataset_name}]")
            if opt_data.get('is_3d', False):
                trainset3d_loader.append(eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "train"))
            else:
                trainset2d_loader.append(eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "train"))
        
        # dataloader for validation
        valid_loader_num = 0
        if len(trainset3d_loader) > 0:
            trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(trainset2d_loader) > 0:
            trainset2d_loader = [MultipleDatasets(trainset2d_loader, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []
        if valid_loader_num > 1:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        num_threads = self.opt['num_threads']
        train_batch_size = self.opt['train']['batch_size']
        self.itr_per_epoch = math.ceil(len(trainset_loader) / train_batch_size / self.world_size) 

        self.sampler = DistributedSampler(trainset_loader)
        self.training_batch_generator = DataLoader(dataset=trainset_loader, batch_size=train_batch_size,
                                          num_workers=num_threads, sampler=self.sampler, 
                                          pin_memory=True, drop_last=False)

    def _make_evaluating_generator(self):
        # data loader and construct batch generator
        # dynamic dataset import
        try:
            opt_data = self.opt['dataset']
            dataset_name = opt_data['name']
            exec(f'from data.{dataset_name} import {dataset_name}')
            globals().update(locals())
        except:
            dataset_list = self.opt['dataset_list']
            for _, opt_data in dataset_list.items():
                dataset_name = opt_data['name']
                exec(f'from data.{dataset_name} import {dataset_name}')
            globals().update(locals())
        
        # self.logger.info(f"Creating dataset ... [{dataset_name}]")
        
        self.testset_loader = eval(dataset_name)(self.opt, opt_data, transforms.ToTensor(), "test")
        
        num_gpus = self.opt['num_gpus']
        num_threads = self.opt['num_threads']
        test_batch_size = self.opt['test']['batch_size']
        self.evaluating_batch_generator = DataLoader(dataset=self.testset_loader, batch_size=test_batch_size,
                                          shuffle=False, num_workers=num_threads, pin_memory=True)

    def _prepare_training(self):
        # prepare network and optimizer
        # if self.local_rank == 0: self.logger.info("Creating network and optimizer ... [seed {}]".format(self.opt['manual_seed']))
        scaler = GradScaler()

        model = get_trainer(self.opt['model']['training'], self.opt, True).to(self.device)
        model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank, 
                                        find_unused_parameters=False, static_graph=True)

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.opt['train']['optim']['lr'], 
                                       weight_decay=self.opt['train']['optim']['weight_decay'])

        # optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=self.opt['train']['optim']['lr'], momentum=0.5, 
        #                                weight_decay=self.opt['train']['optim']['weight_decay'])

        # scheduler = MultiStepLR(optimizer,
        #                         milestones=[(e-1)*self.itr_per_epoch for e in self.opt['train']['optim']['lr_dec_epoch']],
        #                         gamma= 1/self.opt['train']['optim']['lr_dec_factor'])
        
        scheduler = CosineAnnealingLR(optimizer, 
                                      eta_min=self.opt['train']['optim']['lr_min'], 
                                      T_max=self.end_epoch*self.itr_per_epoch)
        
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, 
        #                                             first_cycle_steps = self.itr_per_epoch*self.opt['train']['end_epoch'],
        #                                             cycle_mult = 1.,
        #                                             max_lr = self.opt['train']['optim']['lr'],
        #                                             min_lr = self.opt['train']['optim']['lr_min'],
        #                                             warmup_steps = 50,
        #                                             gamma = 1.,
        #                                             last_epoch = -1)

        start_epoch = 1
        
        # if continue training, load the most recent training state
        if self.local_rank == 0 and self.opt.get('continue_train', False):
            start_epoch, model, optimizer, scheduler, scaler = self.continue_train(model, optimizer, scheduler, scaler)

        self.model = model

        self.start_epoch = start_epoch
        self.end_epoch = self.opt['train']['end_epoch']
        self.save_interval = self.opt['train']['save_interval']

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.apply_grad_clip = self.opt['train']['optim']['apply_grad_clip']
        self.grad_norm = self.opt['train']['optim']['grad_clip_norm']

    def _prepare_testing(self):
        # prepare network
        model = get_trainer(self.opt['model']['training'], self.opt, training=False)
        model = DataParallel(model, device_ids=[0]).cuda()
        
        # # load trained model
        # file_path = osp.join(self.exp_dir, 'training_states', 'epoch_{:02d}.pth.tar'.format(self.opt['test']['epoch']))
        # assert osp.exists(file_path), 'Cannot find training state at ' + file_path
        # ckpt = torch.load(file_path)
        # flag = 0
        # for k, v in ckpt['network'].items():
        #     if k.find('generator.') == -1 and k.find('selector.') == -1:
        #         flag = 1
        #         break
        # if flag==1:
        #     dd = dict()
        #     for k, v in ckpt['network'].items():
        #         if k.find('generator.') == -1 and not k.find('selector.')>-1:
        #             dd[k[:7]+'generator.'+k[7:]] = v
        #     info = model.load_state_dict(dd, strict=False)  # set strict=False due to MANO-related module
        # else:
        #     info = model.load_state_dict(ckpt['network'], strict=False)  # set strict=False due to MANO-related module
        # print(info.unexpected_keys)
        # print(info.missing_keys)
        # self.model = model

        # load trained model
        file_path = osp.join(self.exp_dir, 'training_states', 'epoch_{:02d}.pth.tar'.format(self.opt['test']['epoch']))
        assert osp.exists(file_path), 'Cannot find training state at ' + file_path
        ckpt = torch.load(file_path)
        flag = 0
        for k, v in ckpt['network'].items():
            if k.find('generator.') == -1 and k.find('selector.') == -1:
                flag = 1
                break
        if flag==1:
            dd = dict()
            for k, v in ckpt['network'].items():
                if k.find('generator.') == -1 and not k.find('selector.')>-1:
                    dd[k[:7]+'generator.'+k[7:]] = v
            info = model.load_state_dict(dd, strict=False)  # set strict=False due to MANO-related module
        else:
            info = model.load_state_dict(ckpt['network'], strict=False)  # set strict=False due to MANO-related module
        print(info.unexpected_keys)
        print(info.missing_keys)
        self.model = model

    def save_state(self, epoch):
        os.makedirs(osp.join(self.exp_dir, 'training_states'), exist_ok=True)
        file_path = osp.join(self.exp_dir, 'training_states', 'epoch_{:02d}.pth.tar'.format(epoch))

        state = {
            'network': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch,
        }

        # do not save human model layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smpl_layer' in k or 'mano_layer' in k or 'flame_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        

    def continue_train(self, model: DistributedDataParallel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: GradScaler):
        states_list = glob(osp.join(self.exp_dir, 'training_states', '*.pth.tar'))
        
        # find the most recent training state
        cur_epoch = max([int(file_name[file_name.find('epoch_') + 6:file_name.find('.pth.tar')])
                         for file_name in states_list])

        ckpt_path = osp.join(self.exp_dir, 'training_states', 'epoch_' + '{:02d}'.format(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path)#, map_location=torch.device("cuda", self.local_rank)) 
        
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])

    
        # if self.local_rank == 0:self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        
        return start_epoch, model, optimizer, scheduler, scaler
    
    def start_one_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.model.train()
        for inputs, targets, meta_info in self.training_batch_generator:
            yield (inputs, targets, meta_info)

    def step(self, inputs, targets, meta_info):
        # tracker.track()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = self.model(inputs, targets, meta_info, 'train')

        loss_tensor = [v for v in loss.values() if torch.is_tensor(v)]
        self.optimizer.zero_grad()
        self.scaler.scale(sum(loss_tensor)).backward()
        if self.apply_grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        if self.if_log_grad and dist.get_rank()==0:
            _info = self.log_grad()
        else:
            _info = dict()
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()

        info = self.detach_info(loss)
        info['Info/lr'] = self.scheduler.get_last_lr()[0]
        info.update(_info)
        # tracker.track()

        return info
    
    def evaluate(self, ):
        self.model.eval()
        eval_result = dict()
        with torch.no_grad():
            for inputs, targets, meta_info in tqdm(self.evaluating_batch_generator):
                # tracker.track()
                out = self.model(inputs, targets, meta_info, 'test')
                for k in out.keys():
                    if k not in eval_result.keys():
                        eval_result[k] = []
                    eval_result[k].append(out[k])   # [B, J]
        return {k: torch.cat(v,dim=0) for k, v in eval_result.items()}

    def detach_info(self, loss: dict):
        for k, v in loss.items():
            if isinstance(v, dict):
                loss[k] = self.detach_info(v)
            else:
                loss[k] = v.detach()
        return loss

    def log_grad(self, ):
        grad_backbone = {}
        grad_unfolder = {}
        grad_transformer = {}
        grad_regressor = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                if 'backbone' in name and 'weight' in name: 
                    grad_backbone[name] = torch.norm(p.grad)
                elif 'unfolder' in name and 'weight' in name:
                    grad_unfolder[name] = torch.norm(p.grad)
                elif 'transformer' in name and 'weight' in name: 
                    grad_transformer[name] = torch.norm(p.grad)
                elif 'regressor' in name and 'weight' in name: 
                    grad_regressor[name] = torch.norm(p.grad)

        return {
            "grad":{
                'backbone': grad_backbone,
                'unfolder': grad_unfolder, 
                'transformer': grad_transformer,
                'regressor': grad_regressor
            }
        }