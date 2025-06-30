import numpy as np

def clip_data(vec_data, max_mag):
    """Clip a vector data to set an upper bound to the magnitude. The vector data is of shape (npts, dims), where magnitude is defined as norm(axis=1)"""
    mag = np.linalg.norm(vec_data, axis=1)
    scale = np.minimum(1.0, max_mag / (mag + 1e-12))  # Avoid divide-by-zero
    clipped = vec_data * scale[:, np.newaxis]
    return clipped