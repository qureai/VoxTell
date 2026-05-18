"""
Compare non-compiled vs torch.compiled encoder embeddings for a set of CT volumes.

Two compilation strategies are compared against a baseline:
  - whole:  torch.compile applied to the full encoder as one unit
  - staged: torch.compile applied to each ResidualEncoder stage individually
            (smaller CUDA graphs per stage -> lower peak memory, allows reduce-overhead
             at large input shapes that would OOM with whole-encoder compilation)

For each CT and each requested encoder layer, computes:
  - Mean Absolute Error (MAE)
  - Mean Absolute Relative Error (MARE)
  - Max Absolute Error
  - Cosine similarity

Also reports per-CT inference time for all three modes.

Usage:
    python test_torch_compile_encoder.py \
        --ct_list_file ct_paths.txt \
        --device cuda:0 \
        --model_input_shape 192 192 192 \
        --ct_resize_method trilinear \
        --model_dir /path/to/model \
        --encoder_layer_numbers_list 3 4 5 \
        --compile_mode reduce-overhead
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from voxtell.inference.predictor import VoxTellPredictor

_NO_ALIGN_CORNERS_MODES = {'nearest', 'nearest-exact', 'area'}


class StagedCompiledEncoder(nn.Module):
    """
    Mirrors ResidualEncoder.forward exactly, but each stage (and the stem) is
    compiled separately with torch.compile.

    Compiling per-stage keeps each CUDA graph small, which avoids the OOM that
    occurs when reduce-overhead tries to capture the full encoder in one graph
    at large input shapes (e.g. 256x512x512).
    """

    def __init__(self, encoder: nn.Module, compile_mode: str) -> None:
        super().__init__()
        # Compile stem if present
        self.compiled_stem: Optional[nn.Module] = (
            torch.compile(encoder.stem, mode=compile_mode, fullgraph=False)
            if encoder.stem is not None else None
        )
        # Compile each stage individually
        self.compiled_stages = nn.ModuleList([
            torch.compile(stage, mode=compile_mode, fullgraph=False)
            for stage in encoder.stages
        ])
        self.return_skips: bool = encoder.return_skips

    def forward(self, x: torch.Tensor):
        if self.compiled_stem is not None:
            torch.compiler.cudagraph_mark_step_begin()
            x = self.compiled_stem(x).clone()
        ret = []
        for stage in self.compiled_stages:
            torch.compiler.cudagraph_mark_step_begin()
            x = stage(x).clone()
            ret.append(x)
        return ret if self.return_skips else ret[-1]


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


def run_encoder(encoder, batch: torch.Tensor, device: torch.device, encoder_layer_numbers: List[int]) -> dict:
    with torch.inference_mode():
        skips = encoder(batch.to(device))
    return {layer: skips[layer].float() for layer in encoder_layer_numbers}


def timed_run_encoder(
    encoder,
    batch: torch.Tensor,
    device: torch.device,
    encoder_layer_numbers: List[int],
) -> Tuple[dict, float]:
    """Run encoder and return (embeddings, elapsed_seconds). Uses CUDA events for accurate GPU timing."""
    batch = batch.to(device)
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
        with torch.inference_mode():
            skips = encoder(batch)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> s
    else:
        t0 = time.perf_counter()
        with torch.inference_mode():
            skips = encoder(batch)
        elapsed = time.perf_counter() - t0

    return {layer: skips[layer].float() for layer in encoder_layer_numbers}, elapsed


def compare_embeddings(baseline: torch.Tensor, compiled: torch.Tensor) -> dict:
    """Compute numerical difference metrics between two embedding tensors."""
    a = baseline.cpu()
    b = compiled.cpu()

    abs_diff = (a - b).abs()
    mae = abs_diff.mean().item()
    max_ae = abs_diff.max().item()
    mare = (abs_diff / (a.abs() + 1e-8)).mean().item()

    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    cosine_sim = F.cosine_similarity(a_flat, b_flat, dim=1).mean().item()

    return {'mae': mae, 'mare': mare, 'max_ae': max_ae, 'cosine_sim': cosine_sim}


def run(
    ct_list_file: str,
    device: str,
    model_input_shape: Tuple[int, int, int],
    ct_resize_method: str,
    model_dir: str,
    encoder_layer_numbers: List[int],
    compile_mode: str,
    run_whole: bool = True,
) -> None:
    with open(ct_list_file) as f:
        ct_paths = [line.strip() for line in f if line.strip()]

    n_stages = len(ct_paths)
    print(f"Loaded {n_stages} CT paths")
    print(f"Layers: {encoder_layer_numbers} | Device: {device} | Shape: {model_input_shape} | compile_mode: {compile_mode}\n")

    torch_device = torch.device(device)
    predictor = VoxTellPredictor(model_dir=model_dir, device=torch_device)
    predictor.network = predictor.network.to(torch_device)

    baseline_encoder = predictor.network.encoder
    n_encoder_stages = len(baseline_encoder.stages)

    print(f"Encoder has {n_encoder_stages} stages" +
          (f" + stem" if baseline_encoder.stem is not None else "") + ".\n")

    if run_whole:
        print(f"Compiling whole encoder with mode='{compile_mode}'...")
        whole_compiled_encoder = torch.compile(baseline_encoder, mode=compile_mode, fullgraph=False)
        print("Done.\n")
    else:
        print("Skipping whole-encoder compilation (--no_whole_compile set).\n")

    print(f"Compiling encoder stage-by-stage with mode='{compile_mode}'...")
    staged_compiled_encoder = StagedCompiledEncoder(baseline_encoder, compile_mode=compile_mode)
    staged_compiled_encoder = staged_compiled_encoder.to(torch_device)
    print(f"Done. ({n_encoder_stages} compiled stages" +
          (", 1 compiled stem" if baseline_encoder.stem is not None else "") + ")\n")

    # Accumulators: keyed by (variant, layer)
    variants = (['whole'] if run_whole else []) + ['staged']
    layer_accum: dict = {
        v: {layer: {'mae': [], 'mare': [], 'max_ae': [], 'cosine_sim': []} for layer in encoder_layer_numbers}
        for v in variants
    }
    times: dict = {v: [] for v in ['baseline'] + variants}

    encoders = {'baseline': baseline_encoder}
    if run_whole:
        encoders['whole'] = whole_compiled_encoder
    encoders['staged'] = staged_compiled_encoder

    for i, ct_path in enumerate(ct_paths):
        ct_stem = Path(ct_path).name
        print(f"--- [{i+1}/{n_stages}] {ct_stem} ---")

        try:
            batch = load_and_preprocess_ct(ct_path, model_input_shape, ct_resize_method)

            run_results = {}
            for name, enc in encoders.items():
                embs, t = timed_run_encoder(enc, batch, torch_device, encoder_layer_numbers)
                run_results[name] = embs
                times[name].append(t)

            t_base = times['baseline'][-1]
            t_staged = times['staged'][-1]
            timing_str = f"  time: baseline={t_base:.3f}s"
            if run_whole:
                t_whole = times['whole'][-1]
                timing_str += f"  whole={t_whole:.3f}s ({t_base/t_whole:.2f}x)"
            timing_str += f"  staged={t_staged:.3f}s ({t_base/t_staged:.2f}x)"
            print(timing_str)

            for v in variants:
                print(f"  [{v} vs baseline]")
                for layer in encoder_layer_numbers:
                    metrics = compare_embeddings(run_results['baseline'][layer], run_results[v][layer])
                    shape = run_results['baseline'][layer].shape
                    print(
                        f"    layer {layer} {tuple(shape)}: "
                        f"MAE={metrics['mae']:.6f}  MARE={metrics['mare']:.4%}  "
                        f"MaxAE={metrics['max_ae']:.6f}  CosSim={metrics['cosine_sim']:.8f}"
                    )
                    for k, val in metrics.items():
                        layer_accum[v][layer][k].append(val)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY (mean across all CTs, excluding first CT for timing)")
    print("=" * 80)

    warmup_skip = 1
    if len(times['baseline']) > warmup_skip:
        mean_base = np.mean(times['baseline'][warmup_skip:])
        print(f"\nMean inference time (excluding CT 1):")
        print(f"  Baseline : {mean_base:.3f}s")
        for v in variants:
            mean_v = np.mean(times[v][warmup_skip:])
            print(f"  {v:<8} : {mean_v:.3f}s  ({mean_base/mean_v:.2f}x speedup)")

    for v in variants:
        print(f"\n[{v} vs baseline] embedding differences:")
        header = f"  {'Layer':<8} {'MAE':>12} {'MARE':>12} {'MaxAE':>12} {'CosSim':>14}"
        print(header)
        print("  " + "-" * 60)
        for layer in encoder_layer_numbers:
            acc = layer_accum[v][layer]
            print(
                f"  {layer:<8} "
                f"{np.mean(acc['mae']):>12.6f} "
                f"{np.mean(acc['mare']):>11.4%} "
                f"{np.mean(acc['max_ae']):>12.6f} "
                f"{np.mean(acc['cosine_sim']):>14.8f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare baseline vs torch.compile encoder embeddings.')
    parser.add_argument('--ct_list_file', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model_input_shape', nargs=3, type=int, required=True, metavar=('D', 'H', 'W'))
    parser.add_argument('--ct_resize_method', default='trilinear',
                        choices=['trilinear', 'nearest', 'nearest-exact', 'area'])
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--encoder_layer_numbers_list', nargs='+', type=int, required=True)
    parser.add_argument('--compile_mode', default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--no_whole_compile', action='store_true',
                        help='Skip whole-encoder compilation (useful when it causes OOM)')
    args = parser.parse_args()

    run(
        ct_list_file=args.ct_list_file,
        device=args.device,
        model_input_shape=tuple(args.model_input_shape),
        ct_resize_method=args.ct_resize_method,
        model_dir=args.model_dir,
        encoder_layer_numbers=args.encoder_layer_numbers_list,
        compile_mode=args.compile_mode,
        run_whole=not args.no_whole_compile,
    )


if __name__ == '__main__':
    main()