import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from safetensors.torch import load_file

from monai.data.image_reader import ImageReader
from monai.data.meta_tensor import MetaTensor
from monai.utils import optional_import


class SafeTensorsCTReader(ImageReader):
    """
    MONAI-compatible reader for CT volumes stored in SafeTensors format.

    Expected SafeTensors keys:
        ct_data: Tensor with shape (S, H, W)
            S = slice axis, inferior -> superior
            H = row axis, anterior -> posterior
            W = col axis, patient right -> patient left

        spacing_x: scalar, column spacing
        spacing_y: scalar, row spacing
        spacing_z: scalar, slice spacing

    Optional keys:
        origin_x, origin_y, origin_z:
            RAS-space origin. If absent, origin defaults to 0, 0, 0.

    Output:
        MetaTensor with shape (S, H, W)
        affine maps voxel indices (i, j, k) to RAS world coordinates.
    """

    def __init__(
        self,
        image_key: str = "ct_data",
        spacing_x_key: str = "spacing_x",
        spacing_y_key: str = "spacing_y",
        spacing_z_key: str = "spacing_z",
        origin_keys: Optional[tuple[str, str, str]] = None,
        dtype: torch.dtype = torch.float32,
        dataset_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.image_key = image_key
        self.spacing_x_key = spacing_x_key
        self.spacing_y_key = spacing_y_key
        self.spacing_z_key = spacing_z_key
        self.origin_keys = origin_keys
        self.dtype = dtype
        self.dataset_name = dataset_name

    def verify_suffix(self, filename: str) -> bool:
        filename = str(filename).lower()
        return filename.endswith(".safetensors")

    def read(self, data: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Load the raw SafeTensors dictionary.

        MONAI calls:
            reader.read(filename)
            reader.get_data(raw_obj)
        """
        if isinstance(data, (list, tuple)):
            if len(data) != 1:
                raise ValueError(
                    "SafeTensorsCTReader expects one file per image. "
                    f"Received {len(data)} files."
                )
            data = data[0]

        filename = os.fspath(data)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        return load_file(filename)

    def get_data(self, img: Dict[str, torch.Tensor]):
        if self.image_key not in img:
            raise KeyError(f"Missing image key '{self.image_key}' in SafeTensors file.")

        for key in [self.spacing_x_key, self.spacing_y_key, self.spacing_z_key]:
            if key not in img:
                raise KeyError(f"Missing spacing key '{key}' in SafeTensors file.")

        data = img[self.image_key].to(dtype=self.dtype)

        if data.ndim != 3:
            raise ValueError(
                f"Expected ct_data with shape (S, H, W), got shape {tuple(data.shape)}."
            )

        sx = float(img[self.spacing_x_key].item())
        sy = float(img[self.spacing_y_key].item())
        sz = float(img[self.spacing_z_key].item())

        if self.origin_keys is not None:
            ox_key, oy_key, oz_key = self.origin_keys
            origin_x = float(img[ox_key].item())
            origin_y = float(img[oy_key].item())
            origin_z = float(img[oz_key].item())
        else:
            origin_x = 0.0
            origin_y = 0.0
            origin_z = 0.0

        # Voxel axes:
        #   axis 0 / i: inferior -> superior      => RAS +Z
        #   axis 1 / j: anterior -> posterior     => RAS -Y
        #   axis 2 / k: patient right -> left     => RAS -X
        if self.dataset_name is not None:
            if "lidc" in self.dataset_name.lower():
                # LIDC convention is opposite: axis 0 = superior -> inferior, so we flip the slice axis and negate the slice spacing.
                sz = -sz
        affine = np.array(
            [
                [0.0,  0.0, -sx, origin_x],
                [0.0, -sy,  0.0, origin_y],
                [sz,   0.0, 0.0, origin_z],
                [0.0,  0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        meta = {
            "affine": affine,
            "original_affine": affine.copy(),
            "spatial_shape": np.asarray(data.shape, dtype=np.int16),
            "original_channel_dim": "no_channel",
            "filename_or_obj": "safetensors_ct",
            "space": "RAS",
            "spacing": np.asarray([sz, sy, sx], dtype=np.float32),
        }

        return data, meta