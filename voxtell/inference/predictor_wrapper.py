"""
Batch inference wrapper for VoxTell with multi-GPU support.

Generates per-pixel segmentation scores for a list of CT volumes against a set
of text prompts and dumps the results to disk.

Usage:
    python predictor_wrapper.py \
        --ct_list_file ct_paths.txt \
        --devices cuda:0 cuda:1 \
        --text_prompts "liver" "right kidney" "spleen" \
        --dump_dir /path/to/output \
        --model_dir /path/to/model
"""

import argparse
import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file
from tqdm import tqdm

from voxtell.inference.predictor import VoxTellPredictor


def sanitize_prompt(prompt: str) -> str:
    """Convert a text prompt to a valid directory name."""
    return prompt.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')


def _worker(
    rank: int,
    devices: List[str],
    ct_chunks: List[List[str]],
    text_prompts: List[str],
    text_embeddings: torch.Tensor,
    dump_dir: str,
    model_dir: str,
    use_empty_cache: bool = True,
) -> None:
    """
    Worker function executed on each GPU process.

    Args:
        rank: Process index (used to select device and CT chunk).
        devices: List of device strings (e.g. ['cuda:0', 'cuda:1']).
        ct_chunks: List of CT path lists, one chunk per GPU.
        text_prompts: Text prompts corresponding to each embedding.
        text_embeddings: Precomputed text embeddings of shape (1, P, D) on CPU.
        dump_dir: Root directory for output scores.
        model_dir: Path to VoxTell model directory.
    """
    device = torch.device(devices[rank])
    ct_paths = ct_chunks[rank]

    print(f"[rank {rank}] Initializing predictor on {device} for {len(ct_paths)} volumes.")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)

    # Move shared embeddings to this process's device
    embeddings = text_embeddings.to(device)

    prompt_dirs = [Path(dump_dir) / sanitize_prompt(p) for p in text_prompts]
    for d in prompt_dirs:
        d.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(enumerate(ct_paths), total=len(ct_paths), desc=f"[rank {rank}]", position=rank, leave=True)
    for i, ct_path in pbar:
        ct_path = Path(ct_path)
        # Build output stem: strip .safetensors suffix
        ct_name = ct_path.name
        for suffix in ('.safetensors',):
            if ct_name.endswith(suffix):
                ct_name = ct_name[: -len(suffix)]
                break

        # Skip if all output files already exist
        all_exist = all((d / f"{ct_name}.npz").exists() for d in prompt_dirs)
        if all_exist:
            pbar.set_postfix_str(f"skipped: {ct_path.name}")
            continue

        pbar.set_postfix_str(f"processing: {ct_path.name}")
        try:
            ct_tensors = load_file(str(ct_path))
            if 'ct_data' in ct_tensors:
                img = ct_tensors['ct_data'].numpy()
            else:
                key = next(iter(ct_tensors))
                img = ct_tensors[key].numpy()

            scores = predictor.predict_single_image(
                img,
                text_prompts,
                use_empty_cache=use_empty_cache,
                text_embedding=embeddings,
                return_score=True,
                show_progress=False,
            )  # (P, X, Y, Z) float32

            for j, prompt_dir in enumerate(prompt_dirs):
                out_path = prompt_dir / f"{ct_name}.npz"
                np.savez_compressed(str(out_path), score=scores[j])

        except Exception:
            print(f"[rank {rank}] ERROR processing {ct_path.name}:\n{traceback.format_exc()}")

    print(f"[rank {rank}] Done.")


def run(
    ct_list_file: str,
    devices: List[str],
    text_prompts: List[str],
    dump_dir: str,
    model_dir: str,
    use_empty_cache: bool = True,
) -> None:
    """
    Main entry point for batch inference.

    Generates text embeddings once on the first device, then distributes the CT
    list evenly across all GPUs and runs inference in parallel.

    Args:
        ct_list_file: Path to file containing CT safetensor paths (one per line).
        devices: List of device strings to use.
        text_prompts: Text prompts for segmentation.
        dump_dir: Root output directory.
        model_dir: VoxTell model directory.
    """
    with open(ct_list_file) as f:
        ct_paths = [line.strip() for line in f if line.strip()]

    if not ct_paths:
        raise ValueError(f"No CT paths found in {ct_list_file}")

    print(f"Found {len(ct_paths)} CT volumes, {len(text_prompts)} prompts, {len(devices)} device(s).")

    # Generate text embeddings once on the first device
    print(f"Generating text embeddings on {devices[0]}...")
    embed_predictor = VoxTellPredictor(model_dir=model_dir, device=torch.device(devices[0]))
    text_embeddings = embed_predictor.embed_text_prompts(text_prompts, use_empty_cache=use_empty_cache).cpu()
    del embed_predictor
    torch.cuda.empty_cache()

    # Share across processes
    text_embeddings.share_memory_()

    n_gpus = len(devices)
    # Round-robin split to keep chunks roughly equal
    ct_chunks = [ct_paths[i::n_gpus] for i in range(n_gpus)]

    if n_gpus == 1:
        _worker(0, devices, ct_chunks, text_prompts, text_embeddings, dump_dir, model_dir, use_empty_cache)
    else:
        mp.spawn(
            _worker,
            args=(devices, ct_chunks, text_prompts, text_embeddings, dump_dir, model_dir, use_empty_cache),
            nprocs=n_gpus,
            join=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='VoxTell batch inference: generate per-pixel segmentation scores for CT volumes.'
    )
    parser.add_argument(
        '--ct_list_file', required=True,
        help='Path to a text file with one CT safetensor path per line.',
    )
    parser.add_argument(
        '--devices', nargs='+', required=True,
        help='Devices for inference, e.g. --devices cuda:0 cuda:1',
    )
    parser.add_argument(
        '--text_prompts', nargs='+', required=True,
        help='Text prompts for segmentation, e.g. --text_prompts "liver" "right kidney"',
    )
    parser.add_argument(
        '--dump_dir', required=True,
        help='Root directory for output scores. Scores are saved to <dump_dir>/<prompt>/.',
    )
    parser.add_argument(
        '--model_dir', required=True,
        help='Path to VoxTell model directory containing plans.json and fold_0/checkpoint_final.pth.',
    )
    parser.add_argument(
        '--no_empty_cache', action='store_true', default=False,
        help='Disable torch.cuda.empty_cache() calls between inference steps. '
             'May improve speed at the cost of higher peak GPU memory usage.',
    )
    args = parser.parse_args()

    run(
        ct_list_file=args.ct_list_file,
        devices=args.devices,
        text_prompts=args.text_prompts,
        dump_dir=args.dump_dir,
        model_dir=args.model_dir,
        use_empty_cache=not args.no_empty_cache,
    )


if __name__ == '__main__':
    # Required for mp.spawn on some platforms
    mp.set_start_method('spawn', force=True)
    main()
