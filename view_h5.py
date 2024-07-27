import h5py
import numpy as np


def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Type: {obj.dtype}")
        if obj.shape == ():  # Scalar dataset
            print(f"  Data: {obj[()]}")
        elif obj.size < 10:  # Only print small datasets
            print(f"  Data: {obj[:]}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")


# Open the file
with h5py.File('ann_model_weights.weights.h5', 'r') as f:
    # Print the keys at the root level
    print("Root level keys:", list(f.keys()))

    # Recursively visit all objects in the file
    f.visititems(print_structure)