import copy
import json
import numpy as np
import os
import os.path as osp
import torch
from pycocotools.coco import COCO
from utils.MANO import mano
from utils.visualize import save_obj, seq2video
from utils.preprocessing import load_img, get_bbox, process_bbox, augmentation, process_db_coord_pcf, process_human_model_output
from utils.transforms import world2cam, cam2pixel, transform_joint_to_other_db


class BlurHand(torch.utils.data.Dataset):
    def __init__(self, opt, opt_data, transform, data_split):
        self.opt = opt
        self.opt_params = opt['task_parameters']
        self.transform = transform
        self.data_split = 'train' if data_split == 'train' else 'test'

        self.num_joints = self.opt_params['num_joints']
        self.hm_size = self.opt_params['output_hm_shape']

        # path for images and annotations
        self.img_path = opt_data['img_path']
        self.annot_path = opt_data['annot_path']

        # IH26M-based joint set
        self.joint_set = {'hand': \
                            {'joint_num': 21, # single hand
                            'joints_name': ('Thumb_4', 'Thumb_3', 'Thumb_2', 'Thumb_1',
                                            'Index_4', 'Index_3', 'Index_2', 'Index_1',
                                            'Middle_4', 'Middle_3', 'Middle_2', 'Middle_1',
                                            'Ring_4', 'Ring_3', 'Ring_2', 'Ring_1',
                                            'Pinky_4', 'Pinky_3', 'Pinky_2', 'Pinky_1',
                                            'Wrist'),
                            'flip_pairs': (),
                            'skeleton': ((20,3), (3,2), (2,1), (1,0),
                                         (20,7), (7,6), (6,5), (5,4),
                                         (20,11), (11,10), (10,9), (9,8),
                                         (20,15), (15,14), (14,13), (13,12),
                                         (20,19), (19,18), (18,17), (17,16))
                            }
                        }
        self.joint_set['hand']['joint_type'] = {'right': np.arange(0,self.joint_set['hand']['joint_num']),
                                                'left': np.arange(self.joint_set['hand']['joint_num'],
                                                                  self.joint_set['hand']['joint_num']*2)}
        self.joint_set['hand']['root_joint_idx'] = self.joint_set['hand']['joints_name'].index('Wrist')
        self.datalist = self.load_data()

    def load_data(self):
        # for '*_data.json' files, use pycocotools for fast loading
        db = COCO(osp.join(self.annot_path, self.data_split, f'BlurHand_{self.data_split}_data.json'))

        # otherwise, use standard json protocol
        with open(osp.join(self.annot_path, self.data_split, f'BlurHand_{self.data_split}_MANO_NeuralAnnot.json')) as _f:
            mano_params = json.load(_f)      
        with open(osp.join(self.annot_path, self.data_split, f'BlurHand_{self.data_split}_camera.json')) as _f:
            cameras = json.load(_f)
        with open(osp.join(self.annot_path, self.data_split, f'BlurHand_{self.data_split}_joint_3d.json')) as _f:
            joints = json.load(_f)
        
        datalist = []
        cnt = 0
        for aid in [*db.anns.keys()]:
            ann = db.anns[aid]
            
            # load annotation only if the image_id corresponds to the middle frame of BlurHand
            if not ann['is_middle']:
                continue

            # load each item from annotation
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])
            capture_id = img['capture']
            cam = img['camera']
            frame_idx = img['frame_idx']

            # camera parameters
            t = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32).reshape(3)
            R = np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32).reshape(3,3)
            t = -np.dot(R,t.reshape(3,1)).reshape(3)  # -Rt -> t
            focal = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32).reshape(2)
            princpt = np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32).reshape(2)
            cam_param = {'R': R, 't': t, 'focal': focal, 'princpt': princpt}
           
            # if root is not valid, root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid = np.array(ann['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_valid[self.joint_set['hand']['joint_type']['right']] *= joint_valid[self.joint_set['hand']['root_joint_idx']]
            joint_valid[self.joint_set['hand']['joint_type']['left']] *= joint_valid[self.joint_set['hand']['joint_num'] + \
                                                                                     self.joint_set['hand']['root_joint_idx']]
            # joint coordinates
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam = world2cam(joint_world, R, t)
            joint_cam[np.tile(joint_valid==0, (1,3))] = 1.  # prevent zero division error
            joint_img = cam2pixel(joint_cam, focal, princpt)
            
            # repeat the process for past(1st) image
            aid_past = ann['aid_list'][0]
            ann_past = db.anns[aid_past]
            image_id_past = ann_past['image_id']
            img_past = db.loadImgs(image_id_past)[0]
            capture_id_past = img_past['capture']
            frame_idx_past = img_past['frame_idx']

            joint_valid_past = np.array(ann_past['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_valid_past[self.joint_set['hand']['joint_type']['right']] *= joint_valid_past[self.joint_set['hand']['root_joint_idx']]
            joint_valid_past[self.joint_set['hand']['joint_type']['left']] *= joint_valid_past[self.joint_set['hand']['joint_num'] + \
                                                                                               self.joint_set['hand']['root_joint_idx']]

            joint_world_past = np.array(joints[str(capture_id_past)][str(frame_idx_past)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam_past = world2cam(joint_world_past, R, t)
            joint_cam_past[np.tile(joint_valid_past==0, (1,3))] = 1.
            joint_img_past = cam2pixel(joint_cam_past, focal, princpt)

            # repeat the process for future(5th) image
            aid_future = ann['aid_list'][-1]
            ann_future = db.anns[aid_future]
            image_id_future = ann_future['image_id']
            img_future = db.loadImgs(image_id_future)[0]
            capture_id_future = img_future['capture']
            frame_idx_future = img_future['frame_idx']

            joint_valid_future = np.array(ann_future['joint_valid'],dtype=np.float32).reshape(-1,1)
            joint_valid_future[self.joint_set['hand']['joint_type']['right']] *= joint_valid_future[self.joint_set['hand']['root_joint_idx']]
            joint_valid_future[self.joint_set['hand']['joint_type']['left']] *= joint_valid_future[self.joint_set['hand']['joint_num'] + \
                                                                                                   self.joint_set['hand']['root_joint_idx']]
            
            joint_world_future = np.array(joints[str(capture_id_future)][str(frame_idx_future)]['world_coord'], dtype=np.float32).reshape(-1,3)
            joint_cam_future = world2cam(joint_world_future, R, t)
            joint_cam_future[np.tile(joint_valid_future==0, (1,3))] = 1.
            joint_img_future = cam2pixel(joint_cam_future, focal, princpt)

            # handle the hand_type; ['right', 'left', 'interacting']
            if ann['hand_type'] == 'right':
                hand_type_list = ('right',)
            elif ann['hand_type'] == 'left':
                hand_type_list = ('left',)
            else:
                hand_type_list = ('right', 'left')

            for hand_type in hand_type_list:
                # no avaiable valid joint
                if np.sum(joint_valid[self.joint_set['hand']['joint_type'][hand_type]]) == 0:
                    continue
                
                # process bbox 
                bbox = get_bbox(joint_img[self.joint_set['hand']['joint_type'][hand_type],:2],
                                joint_valid[self.joint_set['hand']['joint_type'][hand_type],0],
                                extend_ratio=1.2)
                bbox = process_bbox(bbox, img_width, img_height, self.opt_params['input_img_shape'])

                # no avaiable bbox
                if bbox is None:
                    continue
                
                # mano parameters for three time steps
                try:
                    mano_param = mano_params[str(capture_id)][str(frame_idx)][hand_type]
                    if mano_param is not None:
                        mano_param['hand_type'] = hand_type
                except KeyError:
                    mano_param = None
                try:
                    mano_param_past = mano_params[str(capture_id_past)][str(frame_idx_past)][hand_type]
                    if mano_param_past is not None:
                        mano_param_past['hand_type'] = hand_type
                except KeyError:
                    mano_param_past = None
                try:
                    mano_param_future = mano_params[str(capture_id_future)][str(frame_idx_future)][hand_type]
                    if mano_param_future is not None:
                        mano_param_future['hand_type'] = hand_type
                except KeyError:
                    mano_param_future = None

                data = {}
                # curremt (middle) frame related
                data['img_path'] = img_path
                data['img_shape'] = (img_height, img_width)
                data['bbox'] = bbox
                data['joint_img'] = joint_img[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_cam'] = joint_cam[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_valid'] = joint_valid[self.joint_set['hand']['joint_type'][hand_type],:]
                data['cam_param'] = cam_param
                data['mano_param'] = mano_param
                data['hand_type'] = hand_type
                data['orig_hand_type'] = ann['hand_type']

                # past frame related
                data['mano_param_past'] = mano_param_past
                data['joint_img_past'] = joint_img_past[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_cam_past'] = joint_cam_past[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_valid_past'] = joint_valid_past[self.joint_set['hand']['joint_type'][hand_type],:]

                # future frame related
                data['mano_param_future'] = mano_param_future
                data['joint_img_future'] = joint_img_future[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_cam_future'] = joint_cam_future[self.joint_set['hand']['joint_type'][hand_type],:]
                data['joint_valid_future'] = joint_valid_future[self.joint_set['hand']['joint_type'][hand_type],:]

                # for SSL
                data['ssl_flag'] = np.random.rand() < 2
                if data['ssl_flag']: cnt += 1
                datalist.append(data)

        print("Total number of sample in BlurHand: {}".format(len(datalist)))
        print("Total number of SL sample: {}, {}%".format(cnt, cnt/len(datalist)*100))
        return datalist
    
    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox, hand_type = data['img_path'], data['img_shape'], data['bbox'], data['hand_type']
        
        img = load_img(img_path)
        img_full = img
        data['cam_param']['t'] /= 1000 # milimeter to meter

        # enforce flip when left hand to make it right hand
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split,
                                                                     self.opt_params['input_img_shape'],
                                                                     enforce_flip=(hand_type=='left'))
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split != 'train':
            do_flip = False

        # mano parameters for middle
        mano_param = data['mano_param']
        if mano_param is not None:
            # mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mano_joint_cam_orig, mano_mesh_cam_orig, mano_joint_img_orig = \
            mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mesh_cam = \
                process_human_model_output(mano_param, data['cam_param'], do_flip,
                                           img_shape, img2bb_trans, rot, 'mano',
                                           self.opt_params['input_img_shape'], self.opt_params['bbox_3d_size'])
            mano_joint_valid = np.ones((mano.joint_num, 1), dtype=np.float32)
            mano_pose_valid = np.ones((mano.orig_joint_num * 3), dtype=np.float32)
            mano_shape_valid = float(True)
        else: # just fill with dummy values
            mano_joint_img = np.zeros((mano.joint_num, 3), dtype=np.float32)
            mano_joint_cam = np.zeros((mano.joint_num, 3), dtype=np.float32)
            mano_joint_trunc = np.zeros((mano.joint_num, 1), dtype=np.float32)
            mano_pose = np.zeros((mano.orig_joint_num * 3), dtype=np.float32) 
            mano_shape = np.zeros((mano.shape_param_dim), dtype=np.float32)
            mano_joint_valid = np.zeros((mano.joint_num, 1), dtype=np.float32)
            mano_pose_valid = np.zeros((mano.orig_joint_num * 3), dtype=np.float32)
            mano_shape_valid = float(False)
            mesh_cam = np.zeros((mano.vertex_num, 3), dtype=np.float32)

        # prepare data for training
        if self.data_split == 'train' or self.data_split == 'test':
            joint_cam = data['joint_cam']
            joint_cam_past = data['joint_cam_past']
            joint_cam_future = data['joint_cam_future']            
            
            # root-relative joint in camera coordinates, milimeter to meter.
            joint_cam = (joint_cam - joint_cam[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000 
            joint_cam_past = (joint_cam_past - joint_cam_past[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000
            joint_cam_future = (joint_cam_future - joint_cam_future[self.joint_set['hand']['root_joint_idx'],None,:]) / 1000

            joint_img = data['joint_img']
            joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1)            
            joint_img_past = data['joint_img_past']
            joint_img_past = np.concatenate((joint_img_past[:,:2], joint_cam_past[:,2:]),1)
            joint_img_future = data['joint_img_future']
            joint_img_future = np.concatenate((joint_img_future[:,:2], joint_cam_future[:,2:]),1)

            joint_img_past, joint_img, joint_img_future, joint_cam_past, joint_cam, joint_cam_future, joint_valid_past, joint_valid, joint_valid_future, joint_trunc_past, joint_trunc, joint_trunc_future = \
            process_db_coord_pcf(joint_img_past, joint_img, joint_img_future,
                                 joint_cam_past, joint_cam, joint_cam_future,
                                 data['joint_valid_past'], data['joint_valid'], data['joint_valid_future'],
                                 do_flip, img_shape, self.joint_set['hand']['flip_pairs'], img2bb_trans,
                                 rot, self.joint_set['hand']['joints_name'], mano.joints_name,
                                 self.opt_params['input_img_shape'], self.opt_params['bbox_3d_size'])

            # mano parameters for past
            mano_param_past = data['mano_param_past']
            if mano_param_past is not None:
                # mano_joint_img_past, mano_joint_cam_past, mano_joint_trunc_past, mano_pose_past, mano_shape_past, mano_joint_cam_orig_past, mano_mesh_cam_orig_past, _ = \
                mano_joint_img_past, mano_joint_cam_past, mano_joint_trunc_past, mano_pose_past, mano_shape_past, mesh_cam_past = \
                    process_human_model_output(mano_param_past, data['cam_param'], do_flip,
                                               img_shape, img2bb_trans, rot, 'mano',
                                               self.opt_params['input_img_shape'], self.opt_params['bbox_3d_size'])
                mano_joint_valid_past = np.ones((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid_past = np.ones((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid_past = float(True)
            else:  # dummy values
                mano_joint_img_past = np.zeros((mano.joint_num,3), dtype=np.float32)
                mano_joint_cam_past = np.zeros((mano.joint_num,3), dtype=np.float32)
                mano_joint_trunc_past = np.zeros((mano.joint_num,1), dtype=np.float32)
                mano_pose_past = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
                mano_shape_past = np.zeros((mano.shape_param_dim), dtype=np.float32)
                mano_joint_valid_past = np.zeros((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid_past = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid_past = float(False)
                mesh_cam_past = np.zeros((mano.vertex_num, 3), dtype=np.float32)
            
            # mano parameters for future
            mano_param_future = data['mano_param_future']
            if mano_param_future is not None:
                # mano_joint_img_future, mano_joint_cam_future, mano_joint_trunc_future, mano_pose_future, mano_shape_future, mano_joint_cam_orig_future, mano_mesh_cam_orig_future, _ = \
                mano_joint_img_future, mano_joint_cam_future, mano_joint_trunc_future, mano_pose_future, mano_shape_future, mesh_cam_future = \
                    process_human_model_output(mano_param_future, data['cam_param'], do_flip,
                                               img_shape, img2bb_trans, rot, 'mano',
                                               self.opt_params['input_img_shape'], self.opt_params['bbox_3d_size'])
                mano_joint_valid_future = np.ones((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid_future = np.ones((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid_future = float(True)
            else:  # dummy values
                mano_joint_img_future = np.zeros((mano.joint_num,3), dtype=np.float32)
                mano_joint_cam_future = np.zeros((mano.joint_num,3), dtype=np.float32)
                mano_joint_trunc_future = np.zeros((mano.joint_num,1), dtype=np.float32)
                mano_pose_future = np.zeros((mano.orig_joint_num*3), dtype=np.float32) 
                mano_shape_future = np.zeros((mano.shape_param_dim), dtype=np.float32)
                mano_joint_valid_future = np.zeros((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid_future = np.zeros((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid_future = float(False)
                mesh_cam_future = np.zeros((mano.vertex_num, 3), dtype=np.float32)

            # target_x, target_y, target_weight = self.generate_sa_simdr(joint_img, joint_trunc)
            # target_x_past, target_y_past, target_weight_past = self.generate_sa_simdr(joint_img_past, joint_trunc_past)
            # target_x_future, target_y_future, target_weight_future = self.generate_sa_simdr(joint_img_future, joint_trunc_future)

            joint_img = np.stack([joint_img_past, joint_img, joint_img_future], 0)
            joint_cam = np.stack([joint_cam_past, joint_cam, joint_cam_future], 0)
            mano_joint_cam = np.stack([mano_joint_cam_past, mano_joint_cam, mano_joint_cam_future], 0)
            mano_pose = np.stack([mano_pose_past, mano_pose, mano_pose_future], 0)
            mano_shape = np.stack([mano_shape_past, mano_shape, mano_shape_future], 0)
            mesh_cam = np.stack([mesh_cam_past, mesh_cam, mesh_cam_future], 0)
            # target_x = np.stack([target_x_past, target_x, target_x_future], 0)
            # target_y = np.stack([target_y_past, target_y, target_y_future], 0)

            joint_valid = np.stack([joint_valid_past, joint_valid, joint_valid_future], 0)
            joint_trunc = np.stack([joint_trunc_past, joint_trunc, joint_trunc_future], 0)
            mano_joint_valid = np.stack([mano_joint_trunc_past, mano_joint_trunc, mano_joint_trunc_future], 0)
            mano_pose_valid = np.stack([mano_pose_valid_past, mano_pose_valid, mano_pose_valid_future], 0)
            mano_shape_valid = np.stack([mano_shape_valid_past, mano_shape_valid, mano_shape_valid_future], 0)
            # target_weight = np.stack([target_weight_past, target_weight, target_weight_future], 0)


            inputs = {'img': img, 'img_path': img_path}
            targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'mano_joint_cam': mano_joint_cam, 
                       'mano_pose': mano_pose, 'mano_shape': mano_shape, 'mesh_cam': mesh_cam}
            meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_valid': mano_joint_valid, 
                         'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 
                        #  'mano_pose_orig': mano_pose_orig, 'mano_shape_orig': mano_shape_orig, 'mano_transl_orig': mano_transl_orig, 
                        #  'K': K_intrinsic, 'affine': affine, 'R': R, 't': t,
                         'hand_type': 1. if data['hand_type']=='right' else 0., 
                         'is_3D': float(True)}
        
        # prepare data for testing
        else:
            # inputs = {'img': img, 'img_path': img_path}
            # targets = {'mano_pose': mano_pose, 'mano_shape': mano_shape}
            # meta_info = {'bb2img_trans': bb2img_trans, 'hand_type': 1. if data['hand_type']=='right' else 0.}
            raise KeyError

        return inputs, targets, meta_info
    
    def __len__(self):
        return len(self.datalist)

def img2hm(joint_img, hm_shape):
    # joint_img: [T, J, 3]
    # hm_shape: [H, W, D]
    joint_xy = joint_img[...,:2]
    H, W, _ = hm_shape
    SIGMA = 3
    EPS = 1e-8

    x = np.arange(W)
    y = np.arange(H)
    hm_idx = np.stack(np.meshgrid(x, y), -1)[None, None]    # [T, J, H, W, 2]

    mu = joint_xy[:,:,None,None,:]  # [T, J, H, W, 2]
    hm = 1 / (np.pi*2*SIGMA) * np.exp(-0.5*((hm_idx-mu)**2/SIGMA).sum(axis=-1)) # [T, J, H, W]
    hm = hm / (hm.sum(axis=-1, keepdims=True).sum(axis=-2, keepdims=True)+EPS)  # rescale to 1

    return hm