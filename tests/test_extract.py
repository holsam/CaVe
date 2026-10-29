'''
CaVe: box cropping, centring and edge padding
'''

# Import external dependencies
import numpy as np
from pathlib import Path

# Import internal CaVe objects
from cave.extract import crop_box

# test_crop_centred: the picked voxel lands at box // 2
def test_crop_centred() -> None:
    vol = np.zeros((20, 20, 20), dtype=np.float32)
    vol[10, 8, 12] = 1.0
    box = crop_box(vol, (10, 8, 12), 6)
    assert box.shape == (6, 6, 6) and box[3, 3, 3] == 1.0 and box.sum() == 1.0

# test_crop_pads_at_edge: a box that leaves the volume keeps its size and is filled with the crop mean
def test_crop_pads_at_edge() -> None:
    vol = np.full((20, 20, 20), 2.0, dtype=np.float32)
    box = crop_box(vol, (1, 10, 10), 8)
    assert box.shape == (8, 8, 8) and np.all(box == 2.0)
