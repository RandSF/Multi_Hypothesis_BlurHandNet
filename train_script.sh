CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 src/train_bh.py --opt options/train/baseline.yml
# if you want to train the selection model, please uncomment the following line
# CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node=2 src/train_bh.py --opt options/train/baseline_rm.yml