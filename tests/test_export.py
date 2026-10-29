'''
CaVe: export tests, star text and box recommendation
'''

# Import external dependencies
import pytest
from pathlib import Path

# Import internal CaVe objects
from cave.estimate import Estimate
from cave.export import recommend_box, write_particles_star
from cave.io import Tomogram

# TOMO: 8 x 6 x 4 voxels at 5.36 A (bin 4 of 1.34 A), so the voxel centre is (x, y, z) = (4, 3, 2)
TOMO = Tomogram('T', 'T_stack_Vol', Path('r.mrc'), None, 5.36, (4, 6, 8), 1.34, (32, 24, 16))

# _est: an estimate at the given voxel position with the given box
def _est(x: float, y: float, z: float, box_a: float) -> Estimate:
    return Estimate(tomogram='T', x=x, y=y, z=z, pixel_size_a=5.36, radius_a=box_a / 2.5, box_a=box_a, snr=3.0, z_lo=0, z_hi=1)

# test_star_centred: the tomogram centre is the origin and offsets scale by the pixel size
def test_star_centred(tmp_path: Path) -> None:
    path = tmp_path / 'p.star'
    write_particles_star([_est(4, 3, 2, 1000.0), _est(5, 4, 3, 1000.0)], {'T': TOMO}, path)
    lines = path.read_text().splitlines()
    assert lines[0] == 'data_particles'
    assert '_rlnTomoName #1' in lines
    assert lines[-2].split() == ['T', '0.00', '0.00', '0.00', '3.000']
    assert [float(v) for v in lines[-1].split()[1:4]] == pytest.approx([5.36, 5.36, 5.36])

# test_recommend_box: percentile of the box edge, in pixels rounded up to even
def test_recommend_box() -> None:
    ests = [_est(4, 3, 2, a) for a in (1000.0, 1100.0, 1200.0)]
    box_a, box_px = recommend_box(ests, percentile=100.0, extract_angpix=10.72)
    assert box_a == 1200.0
    # 1200 / 10.72 = 111.9, so 112 which is already even
    assert box_px == 112
    assert recommend_box(ests, percentile=0.0, extract_angpix=10.0)[1] == 100
