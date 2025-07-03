CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 src/test_bh.py --opt options/test/baseline_rm.yml
# the following line is for generation model evaluation only
# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 src/test_bh.py --opt options/test/baseline.yml