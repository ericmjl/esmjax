from collections import defaultdict
from typing import Optional

import h5py
import numpy as np


def save_model(param_dict: "dict[str, dict[str, np.ndarray]]", 
            fname: Optional[str] = "data/esm1b.h5",
            ) -> None:
    """A two level mapping, whose innermost values are numpy arrays.
    Saves them into an HDF5 file, preserving the dict structure. Load
    back with `load_model`.

    Args:
        param_dict (dict[str, dict[str, np.ndarray]]): Dict of params to save.
        fname (Optional[str]): Full saved model filename.
    """
    # w- throw error if file already exists
    with h5py.File(f"{fname}", "w-") as f:
        # we group arrays by the layer name (first key)
        for layer_name in param_dict:
            # replace else h5py thinks every '/' is a step in hierarchy
            layer_group = f.create_group(layer_name.replace("/", "-"))

            # save each weight as a dataset grouped by parent layer
            for weight_name in param_dict[layer_name]:
                weight_arr = param_dict[layer_name][weight_name]
                layer_group.create_dataset(weight_name, data=weight_arr)


def load_model(h5_fname: Optional[str] = "data/esm1b.h5",
            ) -> "dict[str, dict[str, np.ndarray]]":
    """Load in the model in the two-level dict as originally
    saved by `save_model`.

    Args:
        h5_fname (Optional[str]): saved filename, originally passed into `save_model`.
    Returns:
        dict[str, dict[str, np.ndarray]]: Restored dict of params
    """
    # if key doesn't exist, initialize value to {}
    param_dict = defaultdict(lambda: {})
    
    # reconstruct the two level dict
    with h5py.File(h5_fname, "r") as f:
        for layer_name in f.keys():
            for weight_name in f[layer_name].keys():
                weight = f[layer_name][weight_name][:]
                # make sure to correct the layer_name for indexing
                param_dict[layer_name.replace("-", "/")][weight_name] = weight
    
    return dict(param_dict)
