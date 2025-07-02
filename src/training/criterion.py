###
# criterion for BH, Supervised Learning with GT labels
###
import torch
import torch.nn as nn
from .losses import CoordLoss, ParamLoss, JRCLoss, DiversityLoss

from utils.MANO import mano
from utils.transforms import transform_joint_to_other_db

class Criterion(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.ROOT_IDX = 0    #* it should have not been set here but I think in most dataset the root_idx is 0
        self.JOINT_NUM = 21

        # losses
        self.coord_loss = CoordLoss()
        # self.coord_loss_order_invariant = CoordLossOrderInvariant()
        self.param_loss = ParamLoss()
        # self.vae_loss = VAELoss()
        self.div_loss = DiversityLoss()
        self.rew_loss = JRCLoss()
        
        # parameters
        self.opt_params = opt['task_parameters']
        self.num_k_select = opt['task_parameters']['num_k_select']
        if opt.get('train', False):
            self.opt_loss = opt['train']['loss']

    def forward(self, preds, targets, meta_info, mode = None):
        if mode == 'train':
            gt_pose = targets['mano_pose'][None]   # [K, B, T, ...]
            gt_shape = targets['mano_shape'][None]   # [K, B, T, ...]
            gt_joint = targets['joint_cam'][None]   # [K, B, T, ...]
            gt_joint_mano = targets['mano_joint_cam'][None]   # [K, B, T, ...]
            gt_joint_2d = targets['joint_img'][None][...,:2]   # [K, B, T, ...]

            valid_pose = meta_info['mano_pose_valid'][None]   # [K, B, T, 48]
            valid_shape = meta_info['mano_shape_valid'][None].unsqueeze(-1)   # [K, B, T, 1]
            valid_joint = meta_info['joint_valid'][None]   # [K, B, T, J, 1]
            valid_joint_mano = meta_info['mano_joint_valid'][None]   # [K, B, T, J, 1]
            valid_joint_2d = meta_info['joint_trunc'][None]   # [K, B, T, J, 1]

            pose_all = preds['pose']
            shape_all = preds['shape']
            joint_proj_all = preds['joint_proj'] # projected from `joint`
            joint_all = preds['joint']
            feats_all = preds['feats']
            # score_all = preds['score']

            # below are not multi-hypo
            joint_hm = preds['joint_hm'] # predicted from heatmap

            ### find the best hypothesis
            #! use mm
            jpe_approx = torch.sum((joint_all.detach() - gt_joint)**2, dim=-1, keepdim=True).sqrt()*1000    # [K, B, T, J, 1]
            nan_flag = ~valid_joint.to(bool).expand_as(jpe_approx)
            jpe_approx[nan_flag] = torch.nan
            mpjpe_approx_all = torch.nanmean(jpe_approx, dim=[-1, -2])  # [K, B, T]
            IDX = torch.argsort(mpjpe_approx_all, dim=0)[..., None]    # [K, B, T, 1]

            pose_sort = torch.gather(pose_all, index=IDX.expand_as(pose_all), dim=0)
            shape_sort = torch.gather(shape_all, index=IDX.expand_as(shape_all), dim=0)
            joint_proj_sort = torch.gather(joint_proj_all, index=IDX[...,None].expand_as(joint_proj_all), dim=0)
            joint_sort = torch.gather(joint_all, index=IDX[...,None].expand_as(joint_all), dim=0 )
            mpjpe_approx_sorted = torch.gather(mpjpe_approx_all, index=IDX[...,0], dim=0)
            feats_sorted = torch.gather(feats_all, index=IDX[...,None].expand_as(feats_all), dim=0)

            # error_approx = torch.nanmean(mpjpe_approx, dim=-1)  # [K, B,]
            # IDX_score = torch.argsort(error_approx, dim=0)[..., None]    # [K, B, 1]
            # score_sort = torch.gather(score_all, index=IDX_score.expand_as(score_all), dim=0)

            K_ = self.num_k_select
            pose = pose_sort[:K_]
            shape = shape_sort[:K_]
            joint_proj = joint_proj_sort[:K_]
            joint = joint_sort[:K_]
            mpjpe_approx = mpjpe_approx_sorted[:K_]
            feats = feats_sorted[:K_]
            
            loss = {}

            loss['Loss/joint'] = self.opt_loss['joint'] * self.coord_loss(joint, gt_joint, valid_joint).mean()
            loss['Loss/joint_mano'] = self.opt_loss['joint'] * self.coord_loss(joint, gt_joint_mano, valid_joint_mano).mean()
            loss['Loss/joint_proj'] = self.opt_loss['joint_proj'] * self.coord_loss(joint_proj, gt_joint_2d, valid_joint_2d).mean()
            loss['Loss/joint_hm'] = self.opt_loss['joint_hm'] * self.coord_loss(joint_hm, gt_joint_2d[0], valid_joint_2d[0]).mean()
            loss['Loss/mano_pose'] = self.param_loss(pose, gt_pose, valid_pose).mean()
            loss['Loss/mano_shape'] = self.param_loss(shape, gt_shape, valid_shape).mean()
            loss['Loss/diversity'] = self.opt_loss['diversity'] * self.div_loss(torch.cat([feats_sorted[:K_].detach(), feats_sorted[K_:]], dim=0))

            loss['Info/mpjpe'] = mpjpe_approx.nanmean()

            # num_hit, rank, rank_valid = self.get_selection_info_simple(score_sort.detach(), valid_joint)

            # loss_rm = self.rew_loss(score_sort, num_positive=self.num_k_select, valid=rank_valid.float())   
            # for k, v in loss_rm.items():
            #     loss[k] = v.mean()

            # loss['Info_rm/hit_rate'] = num_hit.nanmean()
            # loss['Info_rm/rank'] = rank.nanmean()
            # for i in range(self.num_k_select):
            #     loss[f'Info_rm/hit_rate{i}'] = num_hit[i].nanmean()
            #     loss[f'Info_rm/rank{i}'] = rank[i].nanmean()
        
            return loss
        
        elif mode == 'test':

            # blur_img = (preds['img'].permute(0, 2, 3, 1)*255).to(torch.uint8)[:,:,::-1]

            # ground-truth
            joint_cam_gt = targets['joint_cam']
            mesh_gt = targets['mesh_cam']  # [B, T, V, 3]
            # validation
            BS, T = mesh_gt.shape[:2]
            J = self.JOINT_NUM
            K = self.opt_params['num_k']
            valid_joint = meta_info['joint_valid'].to(torch.bool)[None].expand(K, BS, T, J, 3)   # [B, T, J, 3]
            # process
            j_reg = torch.from_numpy(mano.joint_regressor).to(mesh_gt.device)[None,None, ...].expand(BS,T,-1,-1)
            mesh_root_gt = torch.matmul(j_reg, mesh_gt)[...,self.ROOT_IDX,:].unsqueeze(-2)  # [B, T, 1, 3]
            mesh_gt = mesh_gt - mesh_root_gt
            # joint_cam_gt = joint_cam_gt - mesh_root_gt
            
            # prediction
            mesh_out = preds['mesh']   # [K, B, T, V, 3]
            j_reg = torch.from_numpy(mano.joint_regressor).to(mesh_out.device)[None,None,None, ...].expand(K, BS,T,-1,-1)
            joint_cam_out = torch.matmul(j_reg, mesh_out)  # had been in the joint order of MANO
            root_joint = joint_cam_out[:,:,:, self.ROOT_IDX].unsqueeze(-2)
            mesh_out = mesh_out - root_joint
            joint_cam_out = joint_cam_out - root_joint
            
            hand_type = meta_info['hand_type'].to(torch.bool).unsqueeze(1).unsqueeze(1).unsqueeze(0)    # [K, B, T, J]
            if torch.any(~hand_type):
                # flip the left hands
                # blur_img = blur_img[:,::-1,:]
                joint_cam_out[...,0] = joint_cam_out[...,0] * (1 - 2*(~hand_type))
                mesh_out[...,0] = mesh_out[...,0] * (1 - 2*(~hand_type))


            # calculating ordered MPJPE
            joint_err = torch.sum((joint_cam_out - joint_cam_gt[None])**2, dim=-1).sqrt()*1000    # [K, B, T, J]
            
            # raise NotImplementedError
            # mask the invalid joints
            joint_err[~valid_joint[...,0]] = torch.nan  # drop the last dim

            mpvpe = torch.mean(
                torch.sum((mesh_out - mesh_gt)**2, dim=-1).sqrt()*1000  # [K, B, T, V]
            , dim=-1)   # [K, B, T]

            mpvpe[~meta_info['mano_shape_valid'].to(torch.bool)[None].expand(K, BS, T)] = torch.nan

            # calculatin PA-MPJPE
            pajpe, joint_aligned, gt_joint_centered, joint_centered = compute_pa_jpe(joint_cam_out, joint_cam_gt, valid_joint) 

            # rank the result
            mpjpe = joint_err.nanmean(dim=-1)   # [K, B, T]
            idx_mpjpe = torch.argsort(mpjpe, dim=0)   # [K, B, T]
            joint_err = torch.gather(joint_err, index=idx_mpjpe.unsqueeze(-1).expand_as(joint_err), dim=0)
            mpvpe = torch.gather(mpvpe, index=idx_mpjpe, dim=0) # [K, B, T]
            pampjpe = pajpe.nanmean(dim=-1)   # [K, B, T]
            # idx_pampjpe = torch.argsort(pampjpe, dim=0)   # [K, B, T]
            pajpe = torch.gather(pajpe, index=idx_mpjpe.unsqueeze(-1).expand_as(pajpe), dim=0)

            # # select according to score
            # score = preds['score']
            # reward = torch.softmax(score, dim=-1)[...,1].unsqueeze(-1).expand(-1, -1, 3)    # [K, B, T]
            # idx_reward = torch.argsort(reward, dim=0, descending=True)
            # joint_err_rew = torch.gather(joint_err, index=idx_reward.unsqueeze(-1).expand_as(joint_err), dim=0)[:self.num_k_select]
            # joint_err_rew, _ = torch.sort(joint_err_rew, dim=0)
            # mpjpe = joint_err.nanmean(dim=[-1, -2])   # [K, B,]
            # idx_mpjpe = torch.argsort(mpjpe, dim=0)   # [K, B,]
            # pajpe_rew = torch.gather(pajpe, index=idx_reward.unsqueeze(-1).expand_as(pajpe), dim=0)
            # score_sort = torch.gather(score, index=idx_mpjpe[...,None].expand_as(score), dim=0)
            # num_hit, rank, _  = self.get_selection_info_simple(score_sort, valid_joint)

            # preds = torch.gather(joint_cam_out, index=idx_mpjpe[...,None,None,None].expand_as(joint_cam_out), dim=0)[0]
            # f, p, r = compute_f_score(preds, joint_cam_gt, th=0.005)  # scaler
            # f2, p2, r2 = compute_f_score(preds, joint_cam_gt, th=0.015)  # scaler
            # f3, p2, r2 = compute_f_score(preds, joint_cam_gt, th=0.065)  # scaler


            return {
                'pa_jpe_current': pajpe[0,:,1], 
                'pa_jpe_past': pajpe[0,:,0], 
                'pa_jpe_future': pajpe[0,:,2],

                'jpe_current': joint_err[0,:,1], 
                'jpe_past': joint_err[0,:,0], 
                'jpe_future': joint_err[0,:,2], 
                
                'jpe_current_mean': joint_err[:,:,1].mean(0), 
                'jpe_past_mean': joint_err[:,:,0].mean(0), 
                'jpe_future_mean': joint_err[:,:,2].mean(0), 

                'jpe_current_middle': joint_err[7,:,1], 
                'jpe_past_middle': joint_err[7,:,0], 
                'jpe_future_middle': joint_err[7,:,2], 

                # 'jpe_current_sort': joint_err_rew[0,:,1], 
                # 'jpe_past_sort': joint_err_rew[0,:,0], 
                # 'jpe_future_sort': joint_err_rew[0,:,2], 

                # 'jpe_current_sort_last': joint_err_rew[-1,:,1], 
                # 'jpe_past_sort_last': joint_err_rew[-1,:,0], 
                # 'jpe_future_sort_last': joint_err_rew[-1,:,2], 

                # 'pa_jpe_sort_current': pajpe_rew[0,:,1], 
                # 'pa_jpe_sort_past': pajpe_rew[0,:,0], 
                # 'pa_jpe_sort_future': pajpe_rew[0,:,2],
                
                'jpe_current_worst':joint_err[-1,:,1], 
                'jpe_past_worst': joint_err[-1,:,0], 
                'jpe_future_worst': joint_err[-1,:,2], 

                'mpvpe_current': mpvpe[0,:,1], 
                'mpvpe_past': mpvpe[0,:,0], 
                'mpvpe_future': mpvpe[0,:,2], 

                # 'hit_rate': num_hit.nanmean(0),
                # 'hit_rate+0': num_hit[0],
                # 'hit_rate-1': num_hit[-1],

                # 'f@5': torch.tensor([f]).cuda(),
                # 'f@15': torch.tensor([f2]).cuda(),
                # 'f@65': torch.tensor([f3]).cuda(),
            }
        else:
            raise KeyError
        
    def get_selection_info_simple(self, score, joint_valid):
            reward = score.softmax(-1)[..., 1]   # [K, B, T]
            joint_valid = joint_valid.sum(dim=[-1, -2, -3]) > 0

            ## correct ratio
            K, B, = reward.shape
            idx_err = torch.arange(K)[:, None].expand_as(reward).to(reward.device) # score has been sorted along true error
            KS = self.num_k_select
            rew_sort, idx_rew = torch.sort(reward, descending=True, dim=0)
            num_hit = torch.zeros_like(idx_rew[:KS])
            for k in range(KS):
                num_hit[k] = (idx_rew[k]<KS)

            rank_p = torch.argsort(
                            idx_rew,
                            descending=False, dim=0
                        )
            rank = rank_p[:KS]    # [Ks, B

            rank_valid = (joint_valid[...,0].sum(-1)>0).float()
            return num_hit.float(), rank.float(), rank_valid
    
def compute_pa_jpe(joint: torch.Tensor, gt_joint: torch.Tensor, valid: torch.Tensor):
    # Step 1: Center both joint sets by subtracting the mean joint position
    # [K, B, T, J, 3]
    
    joint_mean = joint.masked_fill(~valid, torch.nan).nanmean(dim=-2, keepdim=True)
    gt_joint_mean = gt_joint.masked_fill(~valid, torch.nan).nanmean(dim=-2, keepdim=True)
    joint_centered = joint - joint_mean
    gt_joint_centered = gt_joint - gt_joint_mean

    # Cross-covariance matrix ( only consider the valid joints, unlike EBH)
    k, bs = valid.shape[0:2]
    # Ccov = torch.zeros(k, bs, 3, 3).to(valid.device)
    valid_j = valid.squeeze()
    Ccov = torch.matmul(joint.masked_fill(~valid, 0).transpose(-1, -2), gt_joint.masked_fill(~valid, 0))
    # for b in range(bs):
    #     Ccov[b] = torch.einsum('kbij, kbik->kbjk')
    #     torch.matmul(joint_centered[b][valid_j[b]].transpose(-2, -1), gt_joint_centered[b][valid_j[b]].float())
    # Ccov = torch.matmul(joint_centered[valid].transpose(-2, -1), gt_joint_centered.float())  #  [B, 3, 3]
    Umat, Smat, Vt = torch.svd(Ccov)
    Rot = torch.matmul(Vt, Umat.transpose(-2, -1))  # Optimal rotation matrix

    # Ensure a right-handed coordinate system (det(R) should be +1)
    determinant = torch.det(Rot)
    Vt[..., -1] *= torch.sign(determinant).unsqueeze(-1)  # Correct reflection if det is negative
    Rot = torch.matmul(Vt, Umat.transpose(-2, -1))

    # Step 3: Apply rotation to the predicted joints and scale to match the ground truth
    joint_aligned = torch.matmul(joint_centered, Rot.transpose(-1, -2))  # [K, B, T, J, 3]

    # import numpy as np
    # np.savetxt('joint_aligned.txt', joint_aligned[0].detach().cpu().numpy(), ); np.savetxt('joint_aligned_gt.txt', gt_joint_centered[0].detach().cpu().numpy(), )

    # Step 4: Calculate PA-MPJPE (Mean Per Joint Position Error after Procrustes alignment)
    # pa_mpjpe = torch.norm(joint_aligned - joint_gt_centered, dim=-1).mean(dim=1)  # Average over joints
    pa_jpe = torch.sum((joint_aligned - gt_joint_centered)**2, dim=-1).sqrt()*1000  # [K, B, T, J]
    pa_jpe[~valid[...,0]] = torch.nan # drop the last dim

    return pa_jpe, joint_aligned, gt_joint_centered, joint_centered

def compute_f_score(joint, gt_joint, th=0.005):
    # [B, T, J, 3]
    d1, d2 = get_closest_dist(gt_joint, joint) # closest dist for each gt point, closest dist for each pred point
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2) / len(d2))  # how many of our predicted points lie close to a gt point?
        precision = float(sum(d < th for d in d1)) / float(len(d1))  # how many of gt points are matched?

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0
    return fscore, precision, recall

def get_closest_dist(p1, p2):
    shape = p1.shape[:-1]
    dist = torch.cdist(p1.flatten(0,1), p2.flatten(0,1)) # [BT, J1, J2]
    min_dist1, _ = dist.min(dim=-1) # [BT, J1]
    min_dist2, _ = dist.min(dim=-2) # [BT, J2]
    return min_dist1.flatten(), min_dist2.flatten()
