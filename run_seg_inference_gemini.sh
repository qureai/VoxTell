#!/bin/bash

# Hardware Accelerations to prevent CPU single-core bottlenecks
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_MODULE_LOADING=LAZY
# Anti-Fragmentation Flag (Crucial for heavy 3D Segmentation models)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python voxtell/inference/segmentation_wrapper_gemini.py \
--input_jsonl /raid27/mohammad.fahim/voxtell_seg_masks/segmed_ct_rate_data.jsonl \
--devices -1 \
--output_dir /raid27/mohammad.fahim/voxtell_seg_masks/segmed_ct_rate \
--model_dir /raid13/mohammad.fahim/repos/CT_seg_repos/VoxTell/ckpts/voxtell_v1.1 \
--num_workers 8 \
--patch_batch_size 4 \
--tile_step_size 0.5 \
--show_progress