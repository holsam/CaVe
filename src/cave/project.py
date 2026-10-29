'''
CaVe: gaussian smoothing and background flattening of projections for estimation
'''

# Import external dependencies
import numpy as np
from scipy import ndimage

# smooth: gaussian smoothing so the minimum does not just pick out single noisy voxels
def smooth(img: np.ndarray, sigma_px: float) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma_px)

# highpass: subtract a wide gaussian background so only ring-scale structure remains
def highpass(img: np.ndarray, sigma_px: float) -> np.ndarray:
    return img - ndimage.gaussian_filter(img, sigma_px)
