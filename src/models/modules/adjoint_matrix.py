import torch
import numpy as np

kinematic_tree_21 = [
    [0, 1, 2, 3, 4 ],   # Thumb
    [0, 5, 6, 7, 8 ],   # Index
    [0, 9, 10,11,12],   # Middle
    [0, 13,14,15,16],   # Ring
    [0, 17,18,19,20],   # Pinky
]

kinematic_tree_16 =[
    [0, 13,14,15],  # Thumb 
    [0, 1, 2, 3 ],  # Index
    [0, 4, 5, 6 ],  # Middle
    [0, 10,11,12],  # Ring
    [0, 7, 8, 9 ],  # Pinky
]

joint_21_to_16 = [
    [0, 0],                         # root
    [1 , 13], [2 , 14], [3 , 15],   # thumb
    [5 , 1 ], [6 , 2 ], [7 , 3 ],   # index
    [9 , 4 ], [10, 5 ], [11, 6 ],   # middle
    [13, 10], [14, 11], [15, 12],   # ring
    [17, 7 ], [18, 8 ], [19, 9 ],   # pinky
]

def build_adjoint_matrix(num_joint=21, k=4, reverse=True):
    assert num_joint==16 or num_joint==21, "`num_joint` mustbe 16 or 21"

    adjoint = torch.eye(num_joint)
    tree = kinematic_tree_21 if num_joint==21 else kinematic_tree_16

    for finger in tree:
        for i in range(len(finger)-1):
            node = finger[i]
            child = finger[i+1]
            adjoint[node, child] = adjoint[child, node] = 1

    matrixs = torch.stack([torch.matrix_power(adjoint, p).clamp_max(1) for p in range(1, k+1)], dim=0)
    if reverse: matrixs = matrixs.flip(0)
    return matrixs.to(torch.bool)

def build_temporal_spatial_adjoint_matrix(k=8, num_joint=21, num_t=3, reverse=True):
    adjoint = torch.eye(num_joint)
    tree = kinematic_tree_21 if num_joint==21 else kinematic_tree_16

    for finger in tree:
        for i in range(len(finger)-1):
            node = finger[i]
            child = finger[i+1]
            adjoint[node, child] = adjoint[child, node] = 1
    
    adjoint = adjoint.repeat(num_t, num_t)    # [3J, 3J]
    matrixs = torch.stack([torch.matrix_power(adjoint, p).clamp_max(1) for p in range(1, k+1)], dim=0)
    if reverse: matrixs = matrixs.flip(0)
    return matrixs.to(torch.bool)

def build_cross_adjoint_matrix(k=4, reverse=True):
    # build adjoint matrix from 21-joint to 16-joint
    adjoint = torch.zeros([16, 21])
    for i,j in joint_21_to_16:
        adjoint[j, i] = 1
    aux_adj = build_adjoint_matrix(num_joint=21, k=k, reverse=reverse)
    matrixs = torch.matmul(adjoint[None], aux_adj.float())
    return matrixs.to(torch.bool)


if __name__ == "__main__":
    from pprint import pprint
    mat = build_adjoint_matrix()
    mat = mat[4].to(torch.int).numpy().tolist()
    print("i:\t", [i for i in range(21)])
    for i in range(21):
        print(f"{i}:\t", mat[i])
