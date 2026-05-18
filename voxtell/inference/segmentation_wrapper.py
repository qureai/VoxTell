"""
Batch segmentation inference for VoxTell with multi-GPU support.

Reads a JSONL file where each line is a JSON object with the fields:

    {
        "image": "/path/to/volume.safetensors",
        "label": ["lung nodule", "pleural effusion", ...],
        "modality": "CT",
        "orientation_code": "RAS"   // optional, defaults to "RAS"
    }

Each CT volume is reoriented to its record's orientation_code via
SafeTensorsCTReader + MONAI's Orientationd, preprocessed via VoxTellPredictor
(crop-to-nonzero + z-score), and segmented with the per-record text prompts.
Per-label binary masks are saved as bit-packed uint8 of shape (ceil(L/8), X, Y, Z).
To unpack: np.unpackbits(arr, axis=0)[:num_labels]

One safetensors file is written per CT under ``output_dir``.

Usage:
    python segmentation_wrapper.py \
        --input_jsonl /path/to/data.jsonl \
        --devices 0 1 \
        --output_dir /path/to/output \
        --model_dir /path/to/model \
        --num_workers 4 \
        --patch_batch_size 8 \
        --show_progress
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# When this file is executed directly (e.g. `python voxtell/inference/segmentation_wrapper.py`)
# Python has no package context, so `voxtell` is not on sys.path. Adding the
# repo root fixes both the voxtell package import and the safetensors_ct_reader import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import monai.transforms
import numpy as np
import torch
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voxtell.inference.safetensors_ct_reader import SafeTensorsCTReader
from voxtell.inference.predictor import VoxTellPredictor


def _ct_stem(ct_path: Path) -> str:
    """Strip known archive suffixes to get a clean CT identifier."""
    name = ct_path.name
    for suffix in ('.safetensors', '.nii.gz', '.nii', '.npz', '.npy', '.safetensor'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ct_path.stem


def _build_monai_loader(orientation_code: str = "RAS") -> monai.transforms.Compose:
    """
    Build a MONAI pipeline that loads a safetensors CT and reorients it.

    SafeTensorsCTReader provides the affine so Orientationd can correctly
    reorder and flip axes to match the requested orientation code.
    Spacing resampling is intentionally omitted — VoxTell handles variable
    spacings internally via sliding-window inference.

    Args:
        orientation_code: Target orientation, e.g. ``"RAS"``.

    Returns:
        MONAI Compose pipeline.  Call with ``{"image": path_str}``; the
        result ``["image"]`` is a MetaTensor of shape ``(1, X, Y, Z)``.
    """
    return monai.transforms.Compose([
        monai.transforms.LoadImaged(
            keys=["image"],
            reader=SafeTensorsCTReader(),
            image_only=False,
        ),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes=orientation_code),
    ])


class CTDataset(Dataset):
    """
    Dataset that reads JSONL records, loads each CT from safetensors,
    reorients to the per-record orientation_code, and returns unnormalized
    float32 tensors together with the record's text prompts.

    Normalization and crop-to-nonzero are deferred to
    ``VoxTellPredictor.preprocess_gpu()`` in the worker.

    Each item is a ``(tensor, ct_stem, labels, valid)`` tuple where
    ``tensor`` has shape ``(1, X, Y, Z)`` in the target orientation and
    ``labels`` is the list of text prompts from the JSONL record.

    Args:
        records: List of JSONL record dicts (keys: ``image``, ``label``,
            optionally ``orientation_code``).
    """

    def __init__(self, records: List[dict]) -> None:
        self.records = records
        # Cache Compose objects keyed by orientation_code so they are built
        # once per unique code rather than once per item.
        self._loader_cache: Dict[str, monai.transforms.Compose] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _get_loader(self, orientation_code: str) -> monai.transforms.Compose:
        if orientation_code not in self._loader_cache:
            self._loader_cache[orientation_code] = _build_monai_loader(orientation_code)
        return self._loader_cache[orientation_code]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[str], bool]:
        record = self.records[idx]
        ct_path = Path(record["image"])
        ct_stem = _ct_stem(ct_path)
        labels: List[str] = record["label"]
        orientation_code: str = record.get("orientation_code", "RAS")

        try:
            loader = self._get_loader(orientation_code)
            result = loader({"image": str(ct_path)})
            # MetaTensor → plain float32 torch.Tensor, shape (1, X, Y, Z)
            img = result["image"].as_tensor().float()
        except Exception as e:
            print(f"Failed to load {ct_path}: {e}")
            return torch.zeros(1, 1, 1, 1), ct_stem, labels, False

        return img, ct_stem, labels, True


def _collate_fn(batch: list) -> tuple:
    """Return the single item directly, bypassing default tensor collation.

    Needed because ``labels`` is a variable-length list of strings that the
    default collate cannot stack into a tensor.
    """
    return batch[0]


def _worker(
    rank: int,
    devices: List[str],
    record_chunks: List[List[dict]],
    output_dir: str,
    model_dir: str,
    num_workers: int,
    show_progress: bool = False,
    patch_batch_size: int = 1,
    skip_existing: bool = True,
) -> None:
    """
    Worker function executed on each GPU process.

    Builds a DataLoader over the assigned JSONL record chunk and runs
    sliding-window segmentation for every volume.  Text embeddings are
    computed lazily and cached per unique prompts set, so if all records
    share the same prompts (the common case) embeddings are computed once.

    Args:
        rank: Process index (selects device and record chunk).
        devices: List of device strings.
        record_chunks: Per-rank lists of JSONL record dicts.
        output_dir: Directory where per-CT safetensors files are written.
        model_dir: Path to VoxTell model directory.
        num_workers: DataLoader worker processes for data loading.
        show_progress: If True, display a tqdm progress bar.
        patch_batch_size: Sliding-window patches per forward pass (default 1).
        skip_existing: If True, skip CTs whose output file already exists.
    """
    device = torch.device(devices[rank])
    records = record_chunks[rank]

    print(f"[rank {rank}] Initializing predictor on {device} for {len(records)} volumes.")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)

    # Cache: tuple(labels) → (1, P, emb_dim) tensor
    # In the typical case all records share one prompts set → one embed call total.
    embedding_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    dataset = CTDataset(records)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(
        dataloader,
        desc=f"[rank {rank}]",
        position=rank,
        leave=True,
        disable=not show_progress,
    )

    for img, ct_stem, labels, valid in pbar:
        if not valid:
            print(f"[rank {rank}] Skipping invalid CT: {ct_stem}")
            continue

        out_path = Path(output_dir) / f"{ct_stem}.safetensors"
        if skip_existing and out_path.exists():
            if show_progress:
                pbar.set_postfix_str(f"skip {ct_stem}")
            continue

        try:
            # Embed text prompts — retrieve from cache or compute
            prompt_key = tuple(labels)
            if prompt_key not in embedding_cache:
                print(f"[rank {rank}] Embedding {len(labels)} prompt(s) for new prompts set.")
                embedding_cache[prompt_key] = predictor.embed_text_prompts(list(labels))
            text_embeddings = embedding_cache[prompt_key]  # (1, P, emb_dim) on device

            # GPU preprocessing: async pinned-memory transfer → crop-to-nonzero → z-score.
            # All on GPU — no numpy roundtrip.
            img_gpu = img.to(device, non_blocking=True)  # (1, X, Y, Z)
            data_tensor, bbox, orig_shape = predictor.preprocess_gpu(img_gpu)
            # data_tensor: (1, X_crop, Y_crop, Z_crop) on device

            # Sliding-window inference → logits (P, X_crop, Y_crop, Z_crop) on device.
            # use_empty_cache=False: avoids two torch.cuda.empty_cache() sync points
            # per volume (would be 516k cache clears across 258k CTs).
            logits = predictor.predict_sliding_window_return_logits(
                data_tensor, text_embeddings, show_progress=False,
                patch_batch_size=patch_batch_size, use_empty_cache=False,
            )  # stays on device

            # Sigmoid + threshold on GPU — (P, X_crop, Y_crop, Z_crop) bool
            masks_cropped_gpu = torch.sigmoid(logits) > 0.5

            # Crop reversion on GPU: scatter cropped masks into full-volume zeros.
            P = masks_cropped_gpu.shape[0]
            masks_full_gpu = torch.zeros([P, *orig_shape], dtype=torch.bool, device=device)
            masks_full_gpu[
                :,
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1],
            ] = masks_cropped_gpu

            # Move to CPU, bit-pack: (num_labels, X, Y, Z) bool → (ceil(L/8), X, Y, Z) uint8
            # To unpack: np.unpackbits(arr, axis=0)[:num_labels]
            masks_full_cpu = masks_full_gpu.to('cpu', non_blocking=True).numpy().astype(np.uint8)
            packed = np.packbits(masks_full_cpu, axis=0)
            save_file(
                {
                    'seg': torch.from_numpy(packed),
                    'num_labels': torch.tensor(P, dtype=torch.int32),
                },
                str(out_path),
            )

            if show_progress:
                pbar.set_postfix_str(f"saved {ct_stem}")
            else:
                print(f"[rank {rank}] Saved {out_path.name}")

        except Exception:
            print(f"[rank {rank}] ERROR on {ct_stem}:\n{traceback.format_exc()}")


def run(
    input_jsonl: str,
    devices: List[str],
    output_dir: str,
    model_dir: str,
    num_workers: int = 4,
    show_progress: bool = False,
    start_idx: int = 0,
    end_idx: int = -1,
    patch_batch_size: int = 1,
    skip_existing: bool = True,
) -> None:
    """
    Main entry point for batch segmentation inference.

    Reads image paths and text prompts from a JSONL file, distributes records
    across GPUs, and spawns one process per GPU.  Each process writes its
    results directly to ``output_dir``; no merging step is required.

    Args:
        input_jsonl: Path to JSONL file (one record per line with keys
            ``image``, ``label``, and optionally ``orientation_code``).
        devices: List of device strings (e.g. ``['cuda:0', 'cuda:1']``).
        output_dir: Directory for per-CT output safetensors files.
        model_dir: VoxTell model directory (contains ``plans.json`` and
            ``fold_0/checkpoint_final.pth``).
        num_workers: DataLoader worker processes per GPU (default: 4).
        show_progress: If True, display per-rank tqdm progress bars.
        start_idx: Start index (inclusive) into the record list (default: 0).
        end_idx: End index (exclusive) into the record list; -1 = end of list
            (default: -1).
        patch_batch_size: Sliding-window patches per forward pass (default 1).
        skip_existing: If True, skip CTs whose output file already exists.
    """
    with open(input_jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"No records found in {input_jsonl}")

    end = end_idx if end_idx != -1 else len(records)
    records = records[start_idx:end]
    print(f"Processing records [{start_idx}:{end}] — {len(records)} volume(s).")

    if not records:
        raise ValueError(f"No records in range [{start_idx}:{end}]")

    all_prompts = {p for r in records for p in r["label"]}
    print(
        f"Found {len(records)} CT volume(s) | {len(all_prompts)} unique prompt(s) | "
        f"{len(devices)} device(s) | num_workers={num_workers} | patch_batch_size={patch_batch_size}"
    )

    n_gpus = len(devices)
    record_chunks = [records[i::n_gpus] for i in range(n_gpus)]

    worker_args = (
        devices, record_chunks, output_dir, model_dir,
        num_workers, show_progress, patch_batch_size, skip_existing,
    )

    if n_gpus == 1:
        _worker(0, *worker_args)
    else:
        mp.spawn(_worker, args=worker_args, nprocs=n_gpus, join=True)

    print(f"Done. Segmentations saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='VoxTell batch segmentation inference with multi-GPU support.'
    )
    parser.add_argument(
        '--input_jsonl', required=True,
        help=(
            'Path to a JSONL file where each line contains: '
            '{"image": "/path/to/vol.safetensors", "label": ["prompt1", ...], '
            '"modality": "CT", "orientation_code": "RAS"}'
        ),
    )
    parser.add_argument(
        '--devices', nargs='+', type=int, required=True,
        help=(
            'GPU indices for inference. Use -1 to select all available GPUs, '
            'or pass specific indices e.g. --devices 0 1 2'
        ),
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='Directory for per-CT output safetensors files.',
    )
    parser.add_argument(
        '--model_dir', required=True,
        help='Path to VoxTell model directory containing plans.json and '
             'fold_0/checkpoint_final.pth.',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes per GPU (default: 4).',
    )
    parser.add_argument(
        '--show_progress', action='store_true', default=False,
        help='Display a tqdm progress bar for each GPU worker (default: off).',
    )
    parser.add_argument(
        '--start_idx', type=int, default=0,
        help='Start index (inclusive) into the record list (default: 0).',
    )
    parser.add_argument(
        '--end_idx', type=int, default=-1,
        help='End index (exclusive) into the record list; -1 means end of list '
             '(default: -1).',
    )
    parser.add_argument(
        '--patch_batch_size', type=int, default=1,
        help=(
            'Number of sliding-window patches processed per forward pass (default: 1). '
            'Increasing to 2–8 improves GPU utilisation but multiplies peak VRAM '
            'usage accordingly.'
        ),
    )
    parser.add_argument(
        '--no_skip_existing', dest='skip_existing', action='store_false', default=True,
        help=(
            'By default, volumes whose output safetensors file already exists are skipped. '
            'Pass --no_skip_existing to reprocess them.'
        ),
    )
    args = parser.parse_args()

    # Resolve device indices → cuda:X strings
    if args.devices == [-1]:
        n_available = torch.cuda.device_count()
        if n_available == 0:
            raise RuntimeError('No CUDA GPUs found.')
        devices = [f'cuda:{i}' for i in range(n_available)]
    else:
        devices = [f'cuda:{i}' for i in args.devices]
    print(f"Using devices: {devices}")

    run(
        input_jsonl=args.input_jsonl,
        devices=devices,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        num_workers=args.num_workers,
        show_progress=args.show_progress,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        patch_batch_size=args.patch_batch_size,
        skip_existing=args.skip_existing,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
