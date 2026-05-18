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
Per-label binary masks are saved as a multi-channel uint8 tensor of shape
``(num_labels, X, Y, Z)`` where channel ``i`` is the binary mask for
``label[i]``.

One safetensors file is written per CT under ``output_dir``, keyed by ``'seg'``.
"""

import argparse
import json
import sys
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import monai.transforms
import numpy as np
import torch
import torch._dynamo
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- AGGRESSIVE OPTIMIZATIONS ---
# 1. Silence all MONAI deprecation warnings (Terminal IO severely bottlenecks Python CPU threads)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Disable Autograd globally (Massive CPU/RAM savings)
torch.set_grad_enabled(False)

# 3. Enable TF32 for L40 Tensor Cores
torch.set_float32_matmul_precision('high')
# --------------------------------

# Hardware Acceleration: Enable Tensor Cores for massive speedups in Matmuls (Conv3d/Linear)
torch.set_float32_matmul_precision('high')

# When this file is executed directly (e.g. `python voxtell/inference/segmentation_wrapper.py`)
# Python has no package context, so `voxtell` is not on sys.path. Adding the
# repo root fixes both the voxtell package import and the safetensors_ct_reader import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from voxtell.inference.safetensors_ct_reader import SafeTensorsCTReader
from voxtell.inference.predictor_gemini import VoxTellPredictor


def packbits_gpu_axis0(masks_gpu: torch.Tensor) -> torch.Tensor:
    """Equivalent to np.packbits(arr, axis=0) but runs blazing fast on GPU without CPU syncs."""
    masks_gpu = masks_gpu.to(torch.uint8)
    P, X, Y, Z = masks_gpu.shape
    pad_len = (8 - (P % 8)) % 8
    
    # F.pad reads from the back (Z, then Y, then X, then P).
    # Tuple format: (Z_left, Z_right, Y_left, Y_right, X_left, X_right, P_left, P_right)
    if pad_len > 0:
        masks_gpu = torch.nn.functional.pad(masks_gpu, (0, 0, 0, 0, 0, 0, 0, pad_len))
    
    masks_reshaped = masks_gpu.view(-1, 8, X, Y, Z)
    shifts = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=masks_gpu.device)
    shifts = shifts.view(1, 8, 1, 1, 1)
    
    # Multiply by powers of 2 and sum to bit-pack
    packed_gpu = (masks_reshaped * shifts).sum(dim=1, dtype=torch.uint8)
    return packed_gpu


def _ct_stem(ct_path: Path) -> str:
    """Strip known archive suffixes to get a clean CT identifier."""
    name = ct_path.name
    for suffix in ('.safetensors', '.nii.gz', '.nii', '.npz', '.npy', '.safetensor'):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return ct_path.stem


def _build_monai_loader(orientation_code: str = "RAS") -> monai.transforms.Compose:
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
    def __init__(self, records: List[dict]) -> None:
        self.records = records
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
            img = result["image"].as_tensor().float()
        except Exception as e:
            print(f"Failed to load {ct_path}: {e}")
            return torch.zeros(1, 1, 1, 1), ct_stem, labels, False

        return img, ct_stem, labels, True


def _collate_fn(batch: list) -> tuple:
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
    tile_step_size: float = 0.5,
) -> None:
    device = torch.device(devices[rank])
    records = record_chunks[rank]

    print(f"[rank {rank}] Initializing predictor on {device} for {len(records)} volumes.")
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    
    # Algorithmic Speedup: Adjust overlap to save computation
    predictor.tile_step_size = tile_step_size

    # # Hardware Acceleration: Compile model for kernel fusion (CUDAGraphs)
    # print(f"[rank {rank}] Compiling model via torch.compile (first volume will take longer)...")
    # torch._dynamo.config.suppress_errors = True
    # predictor.network = torch.compile(predictor.network, mode="reduce-overhead")

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
            continue

        out_path = Path(output_dir) / f"{ct_stem}.safetensors"
        if skip_existing and out_path.exists():
            if show_progress:
                pbar.set_postfix_str(f"skip {ct_stem}")
            continue

        try:
            prompt_key = tuple(labels)
            if prompt_key not in embedding_cache:
                embedding_cache[prompt_key] = predictor.embed_text_prompts(list(labels))
            text_embeddings = embedding_cache[prompt_key]  # (1, P, emb_dim)

            img_gpu = img.to(device, non_blocking=True)  # (1, X, Y, Z)
            data_tensor, bbox, orig_shape = predictor.preprocess_gpu(img_gpu)

            logits = predictor.predict_sliding_window_return_logits(
                data_tensor, text_embeddings, show_progress=False,
                patch_batch_size=patch_batch_size, use_empty_cache=False,
            )

            masks_cropped_gpu = torch.sigmoid(logits) > 0.5
            P = masks_cropped_gpu.shape[0]
            masks_full_gpu = torch.zeros([P, *orig_shape], dtype=torch.bool, device=device)
            masks_full_gpu[
                :,
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1],
            ] = masks_cropped_gpu

            # Bit-pack directly on GPU to bypass CPU synchronization bottleneck
            packed_gpu = packbits_gpu_axis0(masks_full_gpu)
            
            # Asynchronously move the massively shrunken tensor to CPU
            packed_cpu = packed_gpu.to('cpu', non_blocking=True)
            num_labels_tensor = torch.tensor(P, dtype=torch.int32)

            def save_task(output_dict, path):
                try:
                    save_file(output_dict, path)
                except Exception as e:
                    print(f"Error saving {path}: {e}")

            # Fire-and-forget saving (Unblocks the GPU pipeline immediately)
            threading.Thread(
                target=save_task,
                args=({'seg': packed_cpu, 'num_labels': num_labels_tensor}, str(out_path))
            ).start()

            if show_progress:
                pbar.set_postfix_str(f"saving async {ct_stem}")
            else:
                print(f"[rank {rank}] Saving async {out_path.name}")

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
    tile_step_size: float = 0.5,
) -> None:
    with open(input_jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"No records found in {input_jsonl}")

    end = end_idx if end_idx != -1 else len(records)
    records = records[start_idx:end]

    all_prompts = {p for r in records for p in r["label"]}
    print(
        f"Processing {len(records)} CTs | {len(all_prompts)} prompts | "
        f"{len(devices)} GPUs | workers={num_workers} | patch_batch={patch_batch_size} | overlap={1.0 - tile_step_size:.2f}"
    )

    n_gpus = len(devices)
    record_chunks = [records[i::n_gpus] for i in range(n_gpus)]

    worker_args = (
        devices, record_chunks, output_dir, model_dir,
        num_workers, show_progress, patch_batch_size, skip_existing, tile_step_size
    )

    if n_gpus == 1:
        _worker(0, *worker_args)
    else:
        mp.spawn(_worker, args=worker_args, nprocs=n_gpus, join=True)

    print(f"Done. Background saves are finishing. Outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description='VoxTell batch segmentation inference.')
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--devices', nargs='+', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--show_progress', action='store_true', default=False)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--patch_batch_size', type=int, default=1)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false', default=True)
    parser.add_argument('--tile_step_size', type=float, default=0.5, 
                        help="Sliding window overlap (0.5 = 50%, 0.6 = 40%). Larger values heavily speed up inference.")
    
    args = parser.parse_args()

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
        tile_step_size=args.tile_step_size,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()