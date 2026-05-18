python voxtell/inference/segmentation_wrapper.py \
--input_jsonl /raid27/mohammad.fahim/voxtell_seg_masks/segmed_ct_rate_data.jsonl \
--devices -1 \
--output_dir /raid27/mohammad.fahim/voxtell_seg_masks/segmed_ct_rate \
--model_dir /raid13/mohammad.fahim/repos/CT_seg_repos/VoxTell/ckpts/voxtell_v1.1 \
--num_workers 8 \
--patch_batch_size 8 \
--show_progress
