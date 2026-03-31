"""
Batch encoder embedding extraction for VoxTell with multi-GPU support.

Resizes each CT volume to a fixed model input shape, runs it through the VoxTell
encoder, and dumps the feature maps at the requested encoder layers to a single
compressed NumPy archive (.npz).

Output keys follow the pattern:  {ct_stem}_layer{n}
Values are float16 numpy arrays of shape (C, D, H, W).

Usage:
    python encoder_embeddings_wrapper.py \
        --ct_list_file ct_paths.txt \
        --devices cuda:0 cuda:1 \
        --dump_file /path/to/embeddings.npz \
        --model_input_shape 192 192 192 \
        --ct_resize_method trilinear \
        --model_dir /path/to/model \
        --encoder_layer_numbers_list 3 4 5 \
        --batch_size 4 \
        --num_workers 4 \
        --show_progress
"""

import argparse
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voxtell.inference.predictor import VoxTellPredictor


# Resize modes that do not accept align_corners
_NO_ALIGN_CORNERS_MODES = {'nearest', 'nearest-exact', 'area'}


def _ct_stem(ct_path: Path) -> str:
    """Strip known archive suffixes to get a clean CT identifier."""
    name = ct_path.name
    for suffix in ('.safetensors', '.nii.gz', '.nii', '.npz', '.npy'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ct_path.stem


class CTDataset(Dataset):
    """
    Dataset that loads CT volumes from safetensors files, resizes them to a
    target shape, and applies z-score normalization.

    Each item is a ``(tensor, ct_stem)`` pair where ``tensor`` has shape
    ``(1, D, H, W)`` and is ready to be stacked into a batch.

    Args:
        ct_paths: List of safetensors file paths.
        model_input_shape: Target spatial dimensions ``(D, H, W)``.
        ct_resize_method: Torch interpolate mode (e.g. ``'trilinear'``).
    """

    def __init__(
        self,
        ct_paths: List[str],
        model_input_shape: Tuple[int, int, int],
        ct_resize_method: str,
    ) -> None:
        self.ct_paths = ct_paths
        self.model_input_shape = model_input_shape
        self.ct_resize_method = ct_resize_method
        self._align_corners = (
            None if ct_resize_method in _NO_ALIGN_CORNERS_MODES else False
        )

    def __len__(self) -> int:
        return len(self.ct_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, bool]:
        ct_path = Path(self.ct_paths[idx])
        ct_stem = _ct_stem(ct_path)

        try:
            ct_tensors = load_file(str(ct_path))
            img = ct_tensors.get('ct_data', ct_tensors[next(iter(ct_tensors))])
        except Exception as e:
            print(f"Failed to load {ct_path}: {e}")
            img = torch.zeros(1, *self.model_input_shape)
            return img, ct_stem, False

        # Ensure float32 and shape (1, C, D, H, W) for interpolate
        img = img.float()
        if img.ndim == 3:
            img = img.unsqueeze(0)   # (1, D, H, W)
        img = img.unsqueeze(0)       # (1, C, D, H, W)

        # Resize to model input shape
        interp_kwargs = (
            {} if self._align_corners is None
            else {'align_corners': self._align_corners}
        )
        img = F.interpolate(
            img, size=self.model_input_shape,
            mode=self.ct_resize_method, **interp_kwargs,
        )
        img = img.squeeze(0)         # (C, D, H, W)

        # Z-score normalization per volume
        mean = img.mean()
        std = img.std()
        img = (img - mean) / (std + 1e-8)

        return img, ct_stem, True


def _flush_chunk(results: dict, dump_file: str, rank: int, chunk_idx: int) -> None:
    """Save ``results`` to a numbered chunk file and return the path."""
    chunk_path = Path(dump_file).with_suffix(f".rank{rank}.chunk{chunk_idx}.npz")
    np.savez_compressed(str(chunk_path), **results)
    print(f"[rank {rank}] Flushed {len(results)} embeddings to {chunk_path.name}")


def _worker(
    rank: int,
    devices: List[str],
    ct_chunks: List[List[str]],
    model_input_shape: Tuple[int, int, int],
    ct_resize_method: str,
    dump_file: str,
    model_dir: str,
    encoder_layer_numbers: List[int],
    batch_size: int,
    num_workers: int,
    save_every_n_batches: int,
    show_progress: bool = False,
) -> None:
    """
    Worker function executed on each GPU process.

    Builds a DataLoader over the assigned CT chunk, runs batches through the
    VoxTell encoder, and writes per-rank partial results to disk.

    Results are flushed to numbered chunk files every ``save_every_n_batches``
    batches to bound peak memory usage. All chunks are merged into a single
    rank-level partial file at the end.

    Args:
        rank: Process index (selects device and CT chunk).
        devices: List of device strings.
        ct_chunks: Per-rank lists of CT file paths.
        model_input_shape: Spatial size ``(D, H, W)`` to resize CTs to.
        ct_resize_method: Interpolation mode for resizing.
        dump_file: Final output path (used to derive partial file names).
        model_dir: Path to VoxTell model directory.
        encoder_layer_numbers: Encoder layer indices to extract.
        batch_size: Number of CT volumes per forward pass.
        num_workers: DataLoader worker processes for data loading.
        save_every_n_batches: Flush in-memory results to disk after this many
            batches. Lower values reduce peak memory at the cost of more I/O.
        show_progress: If True, display a tqdm progress bar.
    """
    device = torch.device(devices[rank])
    ct_paths = ct_chunks[rank]

    print(f"[rank {rank}] Initializing predictor on {device} for {len(ct_paths)} volumes.")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)

    dataset = CTDataset(ct_paths, model_input_shape, ct_resize_method)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    results: dict = {}
    chunk_idx = 0
    batch_count = 0

    pbar = tqdm(
        dataloader,
        desc=f"[rank {rank}]",
        position=rank,
        leave=True,
        disable=not show_progress,
    )

    for batch_imgs, batch_stems, batch_valid in pbar:
        if show_progress:
            pbar.set_postfix_str(f"batch size: {batch_imgs.shape[0]}")

        try:
            # batch_imgs: (B, C, D, H, W) — already normalized by CTDataset
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            embeddings = predictor.get_encoder_embedding(batch_imgs, encoder_layer_numbers)
            # embeddings: dict {layer_num -> (B, C_l, D_l, H_l, W_l)}

            for layer_num, emb in embeddings.items():
                emb_cpu = emb.cpu().to(torch.float16)
                for b, ct_stem in enumerate(batch_stems):
                    results[f"{ct_stem}_layer{layer_num}"] = emb_cpu[b].numpy()

            for b, ct_stem in enumerate(batch_stems):
                results[f"{ct_stem}_valid"] = np.array(batch_valid[b].item(), dtype=bool)

        except Exception:
            stems = list(batch_stems)
            print(f"[rank {rank}] ERROR on batch {stems}:\n{traceback.format_exc()}")
            for ct_stem in stems:
                results[f"{ct_stem}_valid"] = np.array(False, dtype=bool)

        batch_count += 1
        if batch_count % save_every_n_batches == 0:
            _flush_chunk(results, dump_file, rank, chunk_idx)
            results = {}
            chunk_idx += 1

    # Flush any remaining results
    if results:
        _flush_chunk(results, dump_file, rank, chunk_idx)
        chunk_idx += 1

    # Merge all chunk files into a single rank-level partial file
    partial_path = Path(dump_file).with_suffix(f".rank{rank}.npz")
    merged: dict = {}
    for i in range(chunk_idx):
        chunk_path = Path(dump_file).with_suffix(f".rank{rank}.chunk{i}.npz")
        merged.update(np.load(str(chunk_path)))
        chunk_path.unlink()
    np.savez_compressed(str(partial_path), **merged)
    print(f"[rank {rank}] Saved {len(merged)} embeddings to {partial_path}")


def run(
    ct_list_file: str,
    devices: List[str],
    dump_file: str,
    model_input_shape: Tuple[int, int, int],
    ct_resize_method: str,
    model_dir: str,
    encoder_layer_numbers: List[int],
    batch_size: int,
    num_workers: int,
    save_every_n_batches: int,
    show_progress: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
) -> None:
    """
    Main entry point for batch encoder embedding extraction.

    Distributes the CT list across GPUs, spawns one process per GPU, collects
    per-rank partial files, and merges them into a single compressed NumPy archive.

    Args:
        ct_list_file: Path to a text file with one CT safetensor path per line.
        devices: List of device strings (e.g. ``['cuda:0', 'cuda:1']``).
        dump_file: Output ``.npz`` file path.
        model_input_shape: Spatial dimensions to resize each CT to.
        ct_resize_method: Torch interpolate mode (e.g. ``'trilinear'``).
        model_dir: VoxTell model directory.
        encoder_layer_numbers: Encoder layer indices to extract embeddings from.
        batch_size: Number of CT volumes per encoder forward pass.
        num_workers: DataLoader worker processes per GPU.
        save_every_n_batches: Flush results to disk after this many batches.
        show_progress: If True, display per-rank tqdm progress bars.
        start_idx: Start index (inclusive) into the CT list (default: 0).
        end_idx: End index (exclusive) into the CT list; -1 means end of list (default: -1).
    """
    with open(ct_list_file) as f:
        ct_paths = [line.strip() for line in f if line.strip()]

    if not ct_paths:
        raise ValueError(f"No CT paths found in {ct_list_file}")

    end = end_idx if end_idx != -1 else len(ct_paths)
    ct_paths = ct_paths[start_idx:end]
    print(f"Processing CT indices [{start_idx}:{end}] — {len(ct_paths)} volume(s).")

    if not ct_paths:
        raise ValueError(f"No CT paths in range [{start_idx}:{end}]")

    print(
        f"Found {len(ct_paths)} CT volumes | {len(encoder_layer_numbers)} layer(s) | "
        f"{len(devices)} device(s) | batch_size={batch_size} | num_workers={num_workers}"
    )

    n_gpus = len(devices)
    ct_chunks = [ct_paths[i::n_gpus] for i in range(n_gpus)]

    worker_args = (
        devices, ct_chunks, model_input_shape, ct_resize_method,
        dump_file, model_dir, encoder_layer_numbers,
        batch_size, num_workers, save_every_n_batches, show_progress,
    )

    if n_gpus == 1:
        _worker(0, *worker_args)
    else:
        mp.spawn(_worker, args=worker_args, nprocs=n_gpus, join=True)

    # Merge per-rank partial files into a single output file
    print("Merging partial results...")
    merged = {}
    for rank in range(n_gpus):
        partial_path = Path(dump_file).with_suffix(f".rank{rank}.npz")
        merged.update(np.load(str(partial_path)))
        partial_path.unlink()

    Path(dump_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dump_file, **merged)
    print(f"Saved {len(merged)} embeddings to {dump_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='VoxTell batch encoder embedding extraction.'
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
        '--dump_file', required=True,
        help='Output .npz file path for all embeddings.',
    )
    parser.add_argument(
        '--model_input_shape', nargs=3, type=int, required=True,
        metavar=('D', 'H', 'W'),
        help='Spatial shape to resize each CT to, e.g. --model_input_shape 192 192 192',
    )
    parser.add_argument(
        '--ct_resize_method', default='trilinear',
        choices=['trilinear', 'nearest', 'nearest-exact', 'area'],
        help='Interpolation mode for resizing CTs (default: trilinear).',
    )
    parser.add_argument(
        '--model_dir', required=True,
        help='Path to VoxTell model directory containing plans.json and fold_0/checkpoint_final.pth.',
    )
    parser.add_argument(
        '--encoder_layer_numbers_list', nargs='+', type=int, required=True,
        help='Encoder layer indices to extract, e.g. --encoder_layer_numbers_list 3 4 5',
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Number of CT volumes per encoder forward pass (default: 1).',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes per GPU for data loading (default: 4).',
    )
    parser.add_argument(
        '--save_every_n_batches', type=int, default=128,
        help='Flush in-memory results to disk after this many batches (default: 128).',
    )
    parser.add_argument(
        '--show_progress', action='store_true', default=False,
        help='Display a tqdm progress bar for each GPU worker (default: off).',
    )
    parser.add_argument(
        '--start_idx', type=int, default=0,
        help='Start index (inclusive) into the CT list (default: 0).',
    )
    parser.add_argument(
        '--end_idx', type=int, default=-1,
        help='End index (exclusive) into the CT list; -1 means end of list (default: -1).',
    )
    args = parser.parse_args()

    run(
        ct_list_file=args.ct_list_file,
        devices=args.devices,
        dump_file=args.dump_file,
        model_input_shape=tuple(args.model_input_shape),
        ct_resize_method=args.ct_resize_method,
        model_dir=args.model_dir,
        encoder_layer_numbers=args.encoder_layer_numbers_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_every_n_batches=args.save_every_n_batches,
        show_progress=args.show_progress,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
