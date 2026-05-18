import pydoc
from queue import Queue
from threading import Thread
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction

class VoxTellPredictor:
    def __init__(self, model_dir: str, device: torch.device = torch.device('cuda'),
                 text_encoding_model: str = 'Qwen/Qwen3-Embedding-4B') -> None:
        self.device = device
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        self.normalization = ZScoreNormalization(intensityproperties={})

        self.tile_step_size = 0.5
        self.perform_everything_on_device = True

        self.tokenizer = AutoTokenizer.from_pretrained(text_encoding_model, padding_side='left')
        self.text_backbone = AutoModel.from_pretrained(text_encoding_model).eval()
        self.max_text_length = 8192

        plans = load_json(join(model_dir, 'plans.json'))
        arch_kwargs = plans['configurations']['3d_fullres']['architecture']['arch_kwargs']
        self.patch_size = plans['configurations']['3d_fullres']['patch_size']

        arch_kwargs = dict(**arch_kwargs)
        for required_import_key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
            if arch_kwargs[required_import_key] is not None:
                arch_kwargs[required_import_key] = pydoc.locate(arch_kwargs[required_import_key])

        network = VoxTellModel(
            input_channels=1,
            **arch_kwargs,
            decoder_layer=4,
            text_embedding_dim=2560,
            num_maskformer_stages=5,
            num_heads=32,
            query_dim=2048,
            project_to_decoder_hidden_dim=2048,
            deep_supervision=False
        )

        checkpoint = torch.load(
            join(model_dir, 'fold_0', 'checkpoint_final.pth'),
            map_location=torch.device('cpu'),
            weights_only=False
        )

        if not isinstance(network, OptimizedModule):
            network.load_state_dict(checkpoint['network_weights'])
        else:
            network._orig_mod.load_state_dict(checkpoint['network_weights'])
        
        network.eval()
        self.network = network.to(device)
        self._gaussian_cache: dict = {}

    def preprocess(self, data: np.ndarray, do_crop_to_nonzero: bool = True) -> Tuple[torch.Tensor, List, Tuple[int, ...]]:
        if data.ndim == 3:
            data = data[None] 
        data = data.astype(np.float32)
        original_shape = data.shape[1:]
        if do_crop_to_nonzero:
            data, _, bbox = crop_to_nonzero(data, None)
        else:
            bbox = None
        data = self.normalization.run(data, None)
        data_tensor = torch.from_numpy(data)
        return data_tensor, bbox, original_shape

    def preprocess_gpu(self, data: torch.Tensor) -> Tuple[torch.Tensor, List, Tuple[int, ...]]:
        assert data.ndim == 4 and data.shape[0] == 1, "Expected (1, X, Y, Z)"
        orig_shape: Tuple[int, ...] = tuple(data.shape[1:])

        mask = data[0] != 0
        proj0 = mask.any(2).any(1)
        if not proj0.any():
            bbox = [[0, s] for s in orig_shape]
            cropped = data
        else:
            proj1 = mask.any(2).any(0)
            proj2 = mask.any(1).any(0)
            nz0 = proj0.nonzero(as_tuple=False)
            nz1 = proj1.nonzero(as_tuple=False)
            nz2 = proj2.nonzero(as_tuple=False)
            bbox = [
                [int(nz0[0, 0]), int(nz0[-1, 0]) + 1],
                [int(nz1[0, 0]), int(nz1[-1, 0]) + 1],
                [int(nz2[0, 0]), int(nz2[-1, 0]) + 1],
            ]
            cropped = data[
                :,
                bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1],
            ]

        mean = cropped.mean()
        std = cropped.std()
        normalized = (cropped - mean) / (std + 1e-8)
        return normalized, bbox, orig_shape

    @torch.inference_mode()
    def get_encoder_embedding(
        self,
        data: Union[np.ndarray, torch.Tensor],
        encoder_layer_numbers: List[int],
    ) -> dict[int, torch.Tensor]:
        if isinstance(data, np.ndarray):
            data_tensor, _, _ = self.preprocess(data, do_crop_to_nonzero=False)
            data_tensor = data_tensor.unsqueeze(0).to(self.device)
        else:
            data_tensor = data.to(self.device)

        skips = self.network.encoder(data_tensor)
        return {layer: skips[layer] for layer in encoder_layer_numbers}

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]) -> List[Tuple]:
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(image_size) - 1
            steps = compute_steps_for_sliding_window(image_size[1:], self.patch_size,
                                                     self.tile_step_size)
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.patch_size,
                                                     self.tile_step_size)
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.patch_size)]]))
        return slicers
    
    @torch.inference_mode()
    def embed_text_prompts(self, text_prompts: Union[List[str], str],
                           use_empty_cache: bool = True) -> torch.Tensor:
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)
        self.text_backbone = self.text_backbone.to(self.device)

        text_prompts = wrap_with_instruction(text_prompts)
        text_tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        text_embed = self.text_backbone(**text_tokens)
        embeddings = last_token_pool(text_embed.last_hidden_state, text_tokens['attention_mask'])
        embeddings = embeddings.view(1, n_prompts, -1)
        self.text_backbone = self.text_backbone.to('cpu')
        if use_empty_cache:
            empty_cache(self.device)
        return embeddings

    @torch.inference_mode()
    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor,
        use_empty_cache: bool = True,
        show_progress: bool = True,
        patch_batch_size: int = 1,
    ) -> torch.Tensor:
        if not isinstance(input_image, torch.Tensor):
            raise ValueError(f"input_image must be a torch.Tensor, got {type(input_image)}")
        
        if use_empty_cache:
            empty_cache(self.device)
            
        # OPTIMIZATION: L40 (Ada Lovelace) natively accelerates bfloat16 massively better than float16
        amp_dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        with torch.autocast(self.device.type, dtype=amp_dtype, enabled=True) if self.device.type == 'cuda' else dummy_context():
            data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size,
                                                       'constant', {'value': 0}, True, None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            predicted_logits = self._internal_predict_sliding_window_return_logits(
                data, text_embeddings, slicers, self.perform_everything_on_device,
                use_empty_cache=use_empty_cache,
                show_progress=show_progress,
                patch_batch_size=patch_batch_size,
            )

            if use_empty_cache:
                empty_cache(self.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        text_embeddings: torch.Tensor,
        slicers: List[Tuple],
        do_on_device: bool = True,
        use_empty_cache: bool = True,
        show_progress: bool = True,
        patch_batch_size: int = 1,
    ) -> torch.Tensor:
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(data_tensor, slicer_list, queue, batch_size):
            batch_patches, batch_slicers = [], []
            for slicer in slicer_list:
                patch = data_tensor[slicer][None].contiguous()
                batch_patches.append(patch)
                batch_slicers.append(slicer)
                if len(batch_patches) == batch_size:
                    queue.put((torch.cat(batch_patches, dim=0), batch_slicers))
                    batch_patches, batch_slicers = [], []
            if batch_patches:
                queue.put((torch.cat(batch_patches, dim=0), batch_slicers))
            queue.put('end')

        if use_empty_cache:
            empty_cache(self.device)

        data = data.to(results_device)
        queue = Queue(maxsize=4)
        t = Thread(target=producer, args=(data, slicers, queue, patch_batch_size))
        t.start()

        predicted_logits = torch.zeros(
            (text_embeddings.shape[1], *data.shape[1:]),
            dtype=torch.half,
            device=results_device,
        )
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

        cache_key = (tuple(self.patch_size), str(results_device))
        if cache_key not in self._gaussian_cache:
            self._gaussian_cache[cache_key] = compute_gaussian(
                tuple(self.patch_size),
                sigma_scale=1. / 8,
                value_scaling_factor=10,
                device=results_device,
            )
        gaussian = self._gaussian_cache[cache_key]

        with tqdm(desc=None, total=len(slicers), disable=not show_progress) as pbar:
            while True:
                item = queue.get()
                if item == 'end':
                    queue.task_done()
                    break
                patches, tile_slicers = item
                n = patches.shape[0]
                text_emb_batch = text_embeddings.expand(n, -1, -1)
                
                predictions = self.network(patches, text_emb_batch)
                
                # OPTIMIZATION: IN-PLACE MEMORY ACCUMULATION
                # This prevents PyTorch from allocating hundreds of new tensors,
                # halving the VRAM bandwidth requirement.
                for i, tile_slice in enumerate(tile_slicers):
                    pred = predictions[i]
                    pred.mul_(gaussian)  # In-place multiply
                    predicted_logits[tile_slice].add_(pred)  # In-place add
                    n_predictions[tile_slice[1:]].add_(gaussian)  # In-place add
                
                queue.task_done()
                pbar.update(n)
        queue.join()

        torch.div(predicted_logits, n_predictions, out=predicted_logits)
        return predicted_logits

    def predict_single_image(
        self,
        data: np.ndarray,
        text_prompts: Union[str, List[str]],
        use_empty_cache: bool = True,
        text_embedding: Union[torch.Tensor, None] = None,
        return_score: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        data_tensor, bbox, orig_shape = self.preprocess(data)

        if text_embedding is not None:
            embeddings = text_embedding.to(self.device)
        else:
            embeddings = self.embed_text_prompts(text_prompts, use_empty_cache=use_empty_cache)

        prediction = self.predict_sliding_window_return_logits(
            data_tensor, embeddings, use_empty_cache=use_empty_cache,
            show_progress=show_progress,
        ).to('cpu')

        with torch.no_grad():
            scores = torch.sigmoid(prediction.float())
            prediction = scores if return_score else (scores > 0.5)

        out_dtype = np.float32 if return_score else np.uint8
        segmentation_reverted_cropping = np.zeros(
            [prediction.shape[0], *orig_shape],
            dtype=out_dtype
        )
        segmentation_reverted_cropping = insert_crop_into_image(
            segmentation_reverted_cropping, prediction, bbox
        )

        return segmentation_reverted_cropping


if __name__ == '__main__':
    import napari

    DEFAULT_IMAGE_PATH = "/path/to/your/image.nii.gz"
    DEFAULT_MODEL_DIR = "/path/to/your/model/directory"
    
    image_path = DEFAULT_IMAGE_PATH
    model_dir = DEFAULT_MODEL_DIR
    text_prompts = ["liver", "right kidney", "left kidney", "spleen"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    img, props = NibabelIOWithReorient().read_images([image_path])
    
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    
    viewer = napari.Viewer()
    viewer.add_image(img, name='image')
    for i, prompt in enumerate(text_prompts):
        viewer.add_labels(voxtell_seg[i], name=f'voxtell_{prompt}')
    napari.run()