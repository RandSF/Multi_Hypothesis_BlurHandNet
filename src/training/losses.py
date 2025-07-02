import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from functools import partial

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[...,2:] * is_3D.float()
            loss = torch.cat((loss[...,:2], loss_z),-1)
        return loss

class CoordLossOrderInvariant(nn.Module):
    def __init__(self):
        super(CoordLossOrderInvariant, self).__init__()

    def forward(self, coord_out_e1, coord_out_e2, coord_gt_e1, coord_gt_e2, valid_e1, valid_e2, is_3D=None, return_order=False):
        # hand-wise minimize
        loss1 = (torch.abs(coord_out_e1 - coord_gt_e1) * valid_e1 + torch.abs(coord_out_e2 - coord_gt_e2) * valid_e2).mean(dim=(-1,-2))
        loss2 = (torch.abs(coord_out_e1 - coord_gt_e2) * valid_e2 + torch.abs(coord_out_e2 - coord_gt_e1) * valid_e1).mean(dim=(-1,-2))
        loss_pf = torch.min(loss1, loss2)

        if return_order:
            # 1 if e1 -> e2 else e2 -> e1
            pred_order = (loss1 < loss2).type(torch.FloatTensor).detach().to(coord_out_e1.device)
            return loss_pf, pred_order
        else:
            return loss_pf

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        loss = torch.abs(param_out - param_gt) * valid
        return loss

class CodeBookDiversityLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, x):
        # x: [C, E]
        pairwise = torch.cdist(x[None], x[None], p=1)[0]
        loss = (-pairwise).exp().triu(diagonal=1).mean()
        return loss

class DiversityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pose: torch.Tensor):
        # pose: K, B, T, J
        K, B, T, J, _ = pose.shape
        feat_pose = rearrange(pose, 'k b t j e-> (b t) k (j e)')
        pairwise_dist = torch.cdist(feat_pose, feat_pose, p=2)    # [BT, K, K]
        loss = (-pairwise_dist).exp().triu(diagonal=1).sum()
        cnt = B*T*K*(K-1)/2 if K>1 else 1
        return loss / cnt
    
class DiversityLoss_nojoint(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pose: torch.Tensor, alpha=0.1):
        # pose: K, B, T
        K, B, T, _ = pose.shape
        feat_pose = rearrange(pose, 'k b t e-> (b t) k e')
        pairwise_dist = torch.cdist(feat_pose, feat_pose, p=1)    # [BT, K, K]
        loss = (-pairwise_dist).exp().triu(diagonal=1).sum()
        cnt = B*T*K*(K-1)/2 if K>1 else 1
        return loss / cnt
    
class InfoNCE(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pos, neg, query, valid, tau=0.07):
        # [N, B, ..., E]
        dis_p = (F.cosine_similarity(pos, query, dim=-1) / tau)
        dis_n = (F.cosine_similarity(neg, query, dim=-1) / tau)
        term_pos = dis_p.exp().sum(0, keepdim=True)
        term_neg = dis_n.exp().sum(0, keepdim=True)
        loss = torch.log(term_pos / (term_pos + term_neg))
        return -loss * valid
    
    
class DiscreteKLLoss(nn.Module):
    def __init__(self, num_k=16, sigma=0.5):
        super().__init__()
        self.register_buffer('weight', None, persistent=False)
        weight = torch.arange(num_k)
        weight = torch.softmax(-weight**2/(2*sigma), dim=0)
        self.weight = weight
    def forward(self, pred_dist, idx, valid, ):
        # pred_dist: [K, B, T]
        # idx, rank: [K, B, T]
        weight = torch.gather(self.weight[...,None,None].expand_as(idx), index=idx, dim=0) # [K, B, T]
        kl_div = F.kl_div(pred_dist.log(), weight, reduction='none')
        return kl_div * valid

class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)

class JRCLoss(nn.Module):
    def __init__(self, alpha = 0.75, non_sigmoid = True, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.non_sigmoid = non_sigmoid
        
    def forward(self, score, valid, num_positive):
        '''
        rews: [K, B, T, 2], #!assum that score has been sorted according to the true order
        valid: [K, B, T]
        '''
        # build positive labels
        data_shape = score.shape
        K = data_shape[0]
        P = num_positive
        
        score = score if self.non_sigmoid else F.sigmoid(score)

        # rank = torch.argsort(
        #     torch.argsort(err, dim=0, descending=False),
        #     dim=0, descending=False
        # )   # the rank of each candidate
        # select_idx = (rank<P).long().unsqueeze(-1)   # [K, B, T, 1], if rank>=P, select the negatiave index (0)
        select_idx = torch.tensor([1]*P + [0]*(K-P))[:,None,None,None].expand(*data_shape[:-1], 1).to(score.device)

        vals_calib = score.softmax(dim=-1)  # along head
        vals_rank = score.softmax(dim=0)    # along hypo

        loss_calib = - torch.gather(vals_calib, index=select_idx, dim=-1).log()[...,0]  # [K, B(, 1)]
        loss_rank = - torch.gather(vals_rank, index=select_idx, dim=-1).log()[...,0]

        loss = {
            'Loss_rm/rm': valid*(self.alpha * loss_calib + (1-self.alpha) * loss_rank), 
            'Loss_rm/calib': loss_calib.detach(),
            'Loss_rm/rank': loss_rank.detach(),
            'Info/score_max': vals_calib[...,1].detach().max(dim=0)[0],
            'Info/score_min':vals_calib[...,1].detach().min(dim=0)[0],
        }

        return loss

class JRCLoss_one(nn.Module):
    def __init__(self, alpha = 0.5, non_sigmoid = True, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.non_sigmoid = non_sigmoid
        
    def forward(self, score, valid, num_positive):
        '''
        rews: [K, B, 2], #!assum that score has been sorted according to the true order
        valid: [K, B, ]
        '''
        # build positive labels
        data_shape = score.shape
        K = data_shape[0]
        P = num_positive
        
        score = score if self.non_sigmoid else F.sigmoid(score)

        select_idx = torch.tensor([1]*P + [0]*(K-P))[:,None,None].expand(*data_shape[:-1], 1).to(score.device)

        vals_calib = score.softmax(dim=-1)  # along head
        vals_rank = score.softmax(dim=0)    # along hypo

        loss_calib = - torch.gather(vals_calib, index=select_idx, dim=-1).log()[...,0]  # [K, B(, 1)]
        loss_rank = - torch.gather(vals_rank, index=select_idx, dim=-1).log()[...,0]

        loss = {
            'Loss_rm/rm': valid*(self.alpha * loss_calib + (1-self.alpha) * loss_rank), 
            'Loss_rm/calib': loss_calib.detach(),
            'Loss_rm/rank': loss_rank.detach(),
            'Info/score_max': vals_calib[...,1].detach().max(dim=0)[0],
            'Info/score_min':vals_calib[...,1].detach().min(dim=0)[0],
        }

        return loss
# class JRCLoss(nn.Module):
#     def __init__(self, alpha = 0.5, non_sigmoid = True, **kwargs):
#         super().__init__()
#         self.alpha = alpha
#         self.non_sigmoid = non_sigmoid
        
#     def forward(self, score, mpjpe, valid, num_positive, mpvpe=0, weight=0.1,):
#         '''
#         rews: [K, B, T, 2]
#         err: [K, B, T,] 
#         valid: [K, B, T]
#         '''
#         # build positive labels
#         K, B, T, _ = score.shape
#         P = num_positive
#         err = mpjpe + weight*mpvpe
        
#         score = score if self.non_sigmoid else F.sigmoid(score)

#         rank = torch.argsort(
#             torch.argsort(err, dim=0, descending=False),
#             dim=0, descending=False
#         )   # the rank of each candidate
#         select_idx = (rank<P).long().unsqueeze(-1)   # [K, B, T, 1], if rank>=P, select the negatiave index (0)

#         vals_calib = score.softmax(dim=-1)  # along head
#         vals_rank = score.softmax(dim=0)    # along hypo

#         loss_calib = - torch.gather(vals_calib, index=select_idx, dim=-1).log()[...,0]  # [K, B(, 1)]
#         loss_rank = - torch.gather(vals_rank, index=select_idx, dim=-1).log()[...,0]

#         loss = {
#             'Loss_rm/rm': valid*(self.alpha * loss_calib + (1-self.alpha) * loss_rank), 
#             'Loss_rm/calib': loss_calib.detach(),
#             'Loss_rm/rank': loss_rank.detach(),
#             'Info/score_max': vals_calib[...,1].detach().max(dim=0)[0],
#             'Info/score_min':vals_calib[...,1].detach().min(dim=0)[0],
#         }

#         return loss

class PairwiseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def get_pairwise_comp_probs(self, score, rank, sigma=1):
        batch_s_ij = torch.unsqueeze(score, dim=1) - torch.unsqueeze(score, dim=0)
        batch_p_ij = torch.sigmoid(sigma * batch_s_ij)

        batch_std_diffs = torch.unsqueeze(rank, dim=1) - torch.unsqueeze(rank, dim=0)
        batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)
        batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

        return batch_p_ij, batch_std_p_ij

    def forward(self, score, metric, mask=1,sigma=1):
        gtrank = (-metric).argsort().argsort().float()
        pred,gt = self.get_pairwise_comp_probs(score, gtrank, sigma=sigma)  # [K, K, B]
        rankloss = torch.triu(gt, diagonal=1) * torch.triu(pred, diagonal=1).log() + \
                    (1-torch.triu(gt, diagonal=1)) * (1-torch.triu(pred, diagonal=1)).log()
        rankloss = torch.sum(rankloss, dim=(0, 1))*mask
        return rankloss.mean()
    
class DiscriminativeLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    def forward(self, score, error):
        # score = score.sigmoid()
        rank = torch.argsort(error, dim=-1).argsort(dim=-1) # [K, B]
        label = (rank<4).float()
        # loss = label * torch.log(score) + (1-label) * torch.log(1-score)
        loss = F.binary_cross_entropy_with_logits(input=score, target=label, reduce='none')
        return loss.mean()

class JRCLoss_batchfirst(nn.Module):
    def __init__(self, num_k_select, alpha = 0.5, non_sigmoid = True, **kwargs):
        super().__init__()
        self.num_positive = num_k_select
        self.alpha = alpha
        self.non_sigmoid = non_sigmoid
        
    def _isin(self, pred: torch.Tensor, gt: torch.Tensor):
        assert pred.shape==gt.shape
        assert pred.shape[-1] == self.num_positive
        res = torch.zeros_like(pred)
        for i in range(self.num_positive):
            pred_repl = pred[:, i][:, None].expand_as(pred)
            isin = (pred_repl == gt).sum(dim=-1)    # if pred[:, i] in gt batchwise
            res[:, i] = isin

        return res.float()

    def forward(self, score, mpjpe, valid, mpvpe=0, weight=0.1):
        '''
        rews: [B, K, 2]
        err: [B, K] 
        '''
        # build positive labels
        B, K, _ = score.shape
        P = self.num_positive
        err = mpjpe + weight*mpvpe
        
        score = score if self.non_sigmoid else F.sigmoid(score)

        rank = torch.argsort(
            torch.argsort(err, dim=-1, descending=False),
            dim=-1, descending=False
        )   # the rank of each candidate
        select_idx = (rank<P).long().unsqueeze(-1)   # [B, K, 1], if the rank>=P, select the negatiave index (0)

        vals_calib = score.softmax(dim=-1)
        vals_rank = score.softmax(dim=-2)

        loss_calib = - torch.gather(vals_calib, index=select_idx, dim=-1).log()  # [B, K, 1]
        loss_rank = - torch.gather(vals_rank, index=select_idx, dim=-1).log()
        loss = {
            'Loss_rm/rm': (self.alpha * loss_calib + (1-self.alpha) * loss_rank)*valid, 
            'Loss_rm/calib': loss_calib.detach(),
            'Loss_rm/rank': loss_rank.detach(),
            'Info/score_max': score.detach().max(dim=-1)[0],
            'Info/score_min':score.detach().min(dim=-1)[0],
        }

        return loss
    
import math
class NLLCoordLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coord_out, coord_logvar, coord_gt, valid):
        # [B, T, J, 3]
        log_prob = (
            -((coord_gt - coord_out) ** 2) / (2 * coord_logvar.exp())
            - 0.5*coord_logvar
            - math.log(math.sqrt(2 * math.pi))
        )   # [B, T, J, 3]
        loss = -log_prob * valid
        return loss