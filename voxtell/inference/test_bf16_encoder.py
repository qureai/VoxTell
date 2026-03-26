"""
Compare FP32 vs BF16 encoder embeddings for a set of CT volumes.

For each CT and each requested encoder layer, computes:
  - Mean Absolute Error (MAE)
  - Mean Absolute Relative Error (MARE)
  - Max Absolute Error
  - Cosine similarity (channel-wise mean)

Usage:
    python test_bf16_encoder.py \
        --ct_list_file ct_paths.txt \
        --device cuda:0 \
        --model_input_shape 192 192 192 \
        --ct_resize_method trilinear \
        --model_dir /path/to/model \
        --encoder_layer_numbers_list 3 4 5
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from voxtell.inference.predictor import VoxTellPredictor

_NO_ALIGN_CORNERS_MODES = {'nearest', 'nearest-exact', 'area'}


def load_and_preprocess_ct(
    ct_path: str,
    model_input_shape: Tuple[int, int, int],
    ct_resize_method: str,
) -> torch.Tensor:
    """Load a CT safetensor, resize, and z-score normalise. Returns (1, C, D, H, W) float32."""
    align_corners = None if ct_resize_method in _NO_ALIGN_CORNERS_MODES else False
    interp_kwargs = {} if align_corners is None else {'align_corners': align_corners}

    ct_tensors = load_file(ct_path)
    img = ct_tensors.get('ct_data', ct_tensors[next(iter(ct_tensors))])
    img = img.float()
    if img.ndim == 3:
        img = img.unsqueeze(0)   # (C, D, H, W)
    img = img.unsqueeze(0)       # (1, C, D, H, W)

    img = F.interpolate(img, size=model_input_shape, mode=ct_resize_method, **interp_kwargs)
    img = img.squeeze(0)         # (C, D, H, W)

    mean, std = img.mean(), img.std()
    img = (img - mean) / (std + 1e-8)

    return img.unsqueeze(0)      # (1, C, D, H, W)


def get_embeddings_fp32(
    predictor: VoxTellPredictor,
    batch: torch.Tensor,
    encoder_layer_numbers: List[int],
) -> dict:
    """Run encoder in FP32 (default model dtype, no autocast)."""
    predictor.network.float()
    with torch.inference_mode():
        skips = predictor.network.encoder(batch.to(predictor.device))
    return {layer: skips[layer].float() for layer in encoder_layer_numbers}


def get_embeddings_bf16(
    predictor: VoxTellPredictor,
    batch: torch.Tensor,
    encoder_layer_numbers: List[int],
) -> dict:
    """Run encoder in BF16 via torch.autocast."""
    predictor.network.float()  # keep weights in FP32, autocast handles compute
    with torch.inference_mode(), torch.autocast(device_type=predictor.device.type, dtype=torch.bfloat16):
        skips = predictor.network.encoder(batch.to(predictor.device))
    return {layer: skips[layer].float() for layer in encoder_layer_numbers}


def compare_embeddings(fp32_emb: torch.Tensor, bf16_emb: torch.Tensor) -> dict:
    """Compute numerical difference metrics between two embedding tensors."""
    fp32 = fp32_emb.cpu()
    bf16 = bf16_emb.cpu()

    abs_diff = (fp32 - bf16).abs()
    mae = abs_diff.mean().item()
    max_ae = abs_diff.max().item()

    # Mean Absolute Relative Error: |diff| / (|fp32| + eps)
    mare = (abs_diff / (fp32.abs() + 1e-8)).mean().item()

    # Cosine similarity: flatten spatial dims, compute per-batch cosine sim
    fp32_flat = fp32.reshape(fp32.shape[0], -1)
    bf16_flat = bf16.reshape(bf16.shape[0], -1)
    cosine_sim = F.cosine_similarity(fp32_flat, bf16_flat, dim=1).mean().item()

    return {
        'mae': mae,
        'mare': mare,
        'max_ae': max_ae,
        'cosine_sim': cosine_sim,
    }


def run(
    ct_list_file: str,
    device: str,
    model_input_shape: Tuple[int, int, int],
    ct_resize_method: str,
    model_dir: str,
    encoder_layer_numbers: List[int],
) -> None:
    with open(ct_list_file) as f:
        ct_paths = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(ct_paths)} CT paths")
    print(f"Layers: {encoder_layer_numbers} | Device: {device} | Shape: {model_input_shape}\n")

    torch_device = torch.device(device)
    predictor = VoxTellPredictor(model_dir=model_dir, device=torch_device)
    predictor.network = predictor.network.to(predictor.device)

    # Accumulators for per-layer aggregate stats
    layer_accum: dict = {layer: {'mae': [], 'mare': [], 'max_ae': [], 'cosine_sim': []} for layer in encoder_layer_numbers}

    for ct_path in ct_paths:
        ct_stem = Path(ct_path).name
        print(f"--- {ct_stem} ---")

        try:
            batch = load_and_preprocess_ct(ct_path, model_input_shape, ct_resize_method)

            fp32_embeddings = get_embeddings_fp32(predictor, batch, encoder_layer_numbers)
            bf16_embeddings = get_embeddings_bf16(predictor, batch, encoder_layer_numbers)

            for layer in encoder_layer_numbers:
                metrics = compare_embeddings(fp32_embeddings[layer], bf16_embeddings[layer])
                shape = fp32_embeddings[layer].shape
                print(
                    f"  layer {layer} {tuple(shape)}: "
                    f"MAE={metrics['mae']:.6f}  MARE={metrics['mare']:.4%}  "
                    f"MaxAE={metrics['max_ae']:.6f}  CosSim={metrics['cosine_sim']:.8f}"
                )
                for k, v in metrics.items():
                    layer_accum[layer][k].append(v)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (mean across all CTs)")
    print("=" * 70)
    header = f"{'Layer':<8} {'MAE':>12} {'MARE':>12} {'MaxAE':>12} {'CosSim':>14}"
    print(header)
    print("-" * 70)
    for layer in encoder_layer_numbers:
        acc = layer_accum[layer]
        print(
            f"{layer:<8} "
            f"{np.mean(acc['mae']):>12.6f} "
            f"{np.mean(acc['mare']):>11.4%} "
            f"{np.mean(acc['max_ae']):>12.6f} "
            f"{np.mean(acc['cosine_sim']):>14.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare FP32 vs BF16 encoder embeddings.')
    parser.add_argument('--ct_list_file', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model_input_shape', nargs=3, type=int, required=True, metavar=('D', 'H', 'W'))
    parser.add_argument('--ct_resize_method', default='trilinear',
                        choices=['trilinear', 'nearest', 'nearest-exact', 'area'])
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--encoder_layer_numbers_list', nargs='+', type=int, required=True)
    args = parser.parse_args()

    run(
        ct_list_file=args.ct_list_file,
        device=args.device,
        model_input_shape=tuple(args.model_input_shape),
        ct_resize_method=args.ct_resize_method,
        model_dir=args.model_dir,
        encoder_layer_numbers=args.encoder_layer_numbers_list,
    )


if __name__ == '__main__':
    main()
