"""
I/O utilities for NumPy array files.
"""

import numpy as np


def load_npy(filepath):
    """Load NumPy array from file."""
    return np.load(filepath)


def save_npy(data, filepath):
    """Save NumPy array to file."""
    np.save(filepath, data)
