import os
import os.path as osp
import torch

from utils.options import parse_options, copy_opt_file
from utils.logger import ColorLogger, init_tb_logger
from utils.misc import mkdir_and_rename
from utils.timer import Timer
import torch.distributed as dist
from training import get_runner

def main():
    # load opt and args from yaml
    opt, args = parse_options()

    # os.environ["TORCH_CPP_LOG_LEVEL"] = "DEBUG"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    local_rank = int(os.environ["LOCAL_RANK"])

    args.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    dist.barrier()

    # timers
    tot_timer = Timer()
    gpu_timer = Timer()
    read_timer = Timer()
    print_timer = Timer()
    
    logger = None
    if dist.get_rank() == 0:
        # directories
        exp_dir = osp.join('experiments', opt['name'])
        tb_dir = osp.join('tb_logger', opt['name'])
        if not opt.get('continue_train', False):
            mkdir_and_rename(osp.join(exp_dir))
            mkdir_and_rename(osp.join(tb_dir))
        
        # logger
        logger = ColorLogger(exp_dir, log_name='train_logs.txt')
        eval_logger = ColorLogger(osp.join('experiments', opt['name'], 'results'), log_name='test_log.txt')
        tb_logger = init_tb_logger(tb_dir)
        
        # copy the yml file to the experiment root
        copy_opt_file(args.opt, exp_dir)

    trainer = get_runner(opt['model']['runner'], opt, args, logger, training=True)
    

    tot_iter = (trainer.start_epoch-1) * trainer.itr_per_epoch
    for epoch in range(trainer.start_epoch, (trainer.end_epoch+1)):
        tot_timer.tic()
        read_timer.tic()
        for itr, batch in enumerate(trainer.start_one_epoch(epoch)):
            read_timer.toc()
            gpu_timer.tic()

            info = trainer.step(*batch)
            # backward
            dist.barrier()

            # tensorboard logging
            tot_iter += 1
            gpu_timer.toc()
            print_timer.tic()

            if dist.get_rank()==0:
                grad_info = info.pop('grad', False)
                if grad_info: 
                    for k, v in grad_info.items():
                        tb_logger.add_scalars(f'Grad/{k}', v, tot_iter)   # skip 'modules' and 'predicotr'
                for k,v in info.items(): 
                    if isinstance(v, dict): tb_logger.add_scalars(k, v, tot_iter)
                    else: tb_logger.add_scalar(k, v, tot_iter)
                
                screen_loss_info = get_info_str(info)
                screen_basic_info = [
                    f'Epoch {epoch:d}/{trainer.end_epoch:d} itr {itr+1:d}/{trainer.itr_per_epoch:d}: tot {tot_iter:d}:',
                    f'speed: {tot_timer.average_time:.2f}(c{gpu_timer.average_time:.2f}s p{print_timer.average_time:.2f}s r{read_timer.average_time:.2f}s)s/itr', 
                    f'{(tot_timer.average_time / 3600. * trainer.itr_per_epoch):.2f}h/epoch',
                    ]
                # screen_loss_info = ['%s: %.4f' % (k, v) for k,v in info.items()]
                # screen_basic_info = [
                #     'Epoch %d/%d itr %d/%d: tot %d:' % (epoch, trainer.end_epoch, itr+1, trainer.itr_per_epoch, tot_iter),
                #     'speed: %.2f(c%.2fs p%.2fs r%.2fs)s/itr' % (
                #         tot_timer.average_time, gpu_timer.average_time, print_timer.average_time, read_timer.average_time),
                #     '%.2fh/epoch' % (tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                #     ]
                
                screen = screen_basic_info + screen_loss_info
                trainer.logger.info('\n\t'.join(screen))

            print_timer.toc()
            tot_timer.toc()
            tot_timer.tic()
            read_timer.tic()
        if trainer.local_rank==0 and (epoch%trainer.save_interval==0 or epoch==trainer.end_epoch):
            trainer.save_state(epoch)
            logger.info("Saving training states of epoch {}".format(epoch))
        
        # evaluation
        eval_result = trainer.evaluate()
        eval_result_gather = {}
        for k, v in eval_result.items():
            v_list = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
            dist.gather(v, v_list if dist.get_rank()==0 else None)
            eval_result_gather[k] = torch.cat(v_list, dim=0).nanmean().item()
        if trainer.local_rank==0:
            for k, v in eval_result_gather.items(): tb_logger.add_scalar('Eval/'+k, v, epoch)
            eval_logger.info('evaluation {}_epoch_{}'.format(opt['name'], epoch))
            try:
                eval_logger.info('MPJPE @ CURRENT: %.2f mm' % eval_result_gather['jpe_current'])
            except:
                pass
            try:
                eval_logger.info('MPJPE @ PAST: %.2f mm' % eval_result_gather['jpe_past'])
            except:
                pass
            try:
                eval_logger.info('MPJPE @ FUTURE: %.2f mm' % eval_result_gather['jpe_future'])    
            except:
                pass
            try:
                eval_logger.info('MPVPE @ CURRENT: %.2f mm' % eval_result_gather['mpvpe_current'])
            except:
                pass
            
    dist.destroy_process_group()

def get_info_str(info: dict, prefix=""):
    res = []
    for k, v in info.items():
        if isinstance(v, dict):
            res.extend(get_info_str(v, prefix=k+'/'))
        else:
            res.append(f"{prefix}{k}: {v:.4f}")
    return res

if __name__ == '__main__':
    main()