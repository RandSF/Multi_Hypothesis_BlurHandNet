import torch
import numpy as np
from torch.nn import functional as F
import torchgeometry as tgm

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix
    
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    return rotMat

def rot2aa(mat, eps=1e-8):
    ### mat: [..., 3, 3]
    batch_shape = mat.shape[:-2]
    mat = mat.reshape(-1, 3, 3)
    tr = torch.tensor([torch.trace(m) for m in mat])
    theta = torch.arccos((tr-1)/2)
    sin = torch.sin(theta)

    aa = torch.stack([
        (mat[...,2,1]-mat[...,1,2]) / (2*sin+eps), 
        (mat[...,0,2]-mat[...,2,0]) / (2*sin+eps), 
        (mat[...,1,0]-mat[...,0,1]) / (2*sin+eps), 
    ], dim=-1)
    # print(torch.sum(aa*aa, -1))
    aa = (theta.unsqueeze(-1)*aa).reshape(*batch_shape, 3)
    return aa

def eular2aa(eular):
   # eular: [B, 3]
   B, _ = eular.shape
   rot = batch_euler2matzxy(eular)   # [B, 3, 3]
   rot_mat = torch.cat([rot,torch.zeros((B,3,1)).to(rot.device).float()],2) # 3x4 rotation matrix
   axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
   axis_angle[torch.isnan(axis_angle)] = 0.0
   aa = axis_angle.reshape(B, -1)  # [B, 3]
   return aa

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:,:,0] / (width-1) * 2 - 1
    y = joint_xy[:,:,1] / (height-1) * 2 - 1
    grid = torch.stack((x,y),2)[:,:,None,:]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:,:,:,0] # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0,2,1).contiguous() # batch_size, joint_num, channel_dim
    return img_feat

def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2,3))
    accu_y = heatmap3d.sum(dim=(2,4))
    accu_z = heatmap3d.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out

def soft_argmax_2d(heatmap2d, temperature=1):
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height*width))
    heatmap2d = F.softmax(heatmap2d*temperature, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))

    accu_x = heatmap2d.sum(dim=2)
    accu_y = heatmap2d.sum(dim=3)

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out

def restore_bbox(bbox_center, bbox_size):
    bbox = bbox_center.view(-1,1,2) + torch.cat((-bbox_size.view(-1,1,2)/2., bbox_size.view(-1,1,2)/2.),1) # xyxy in (output_hm_shape[2], output_hm_shape[1]) space
    bbox = bbox.view(-1,4)
    return bbox
