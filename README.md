# Multi-Hypothesis_BlurHandNet
This repository is an official implementation of the paper **"RMulti-Hypothesis 3D Hand Mesh Recovering from a Single Blurry Image**". 
The detailed demo and checkpoint will be coming soon. Currently, training and testing can be done by running
```
source init.sh
source train_script.sh
source test_script.sh
```

there is a known "problem" of `torchgeometry.rotation_matrix_to_quaternion` at line 302-304 of `torchgeometry/core/conversions.py` replace L302-304 by
```
mask_c1 = mask_d2 * (~mask_d0_d1)       # mask_c1 = mask_d2 * (1 - mask_d0_d1)
mask_c2 = (~mask_d2) * mask_d0_nd1      # mask_c2 = (1 - mask_d2) * mask_d0_nd1
mask_c3 = (~mask_d2) * (~mask_d0_nd1)   # mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
```
