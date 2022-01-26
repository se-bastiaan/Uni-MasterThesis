import numpy as np


def tensor2nparr(tensor):
    np_arr = tensor.detach().cpu().numpy()
    np_arr = (np.moveaxis(np_arr, 1, 3) * 255).astype(np.uint8)
    return np_arr


def g(c, window_size=7):
    return max(1, c - int(window_size / 2))
