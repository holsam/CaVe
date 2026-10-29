'''
CaVe: estimation tests on a synthetic ring that exists only in a z range.
'''

# Import external dependencies
import numpy as np, pytest
from pathlib import Path

# Import internal CaVe objects
from cave.config import Config
from cave.estimate import Estimate, estimate_candidate, read_estimates, ring_snr, write_estimates
from cave.pick import Pick

# PX: voxel size of the synthetic volume, so 450 A is 84 voxels
PX = 5.36

# _config: coarse sweep and short walk to keep the test fast
def _config() -> Config:
    return Config(raw_dir=Path('.'), tomograms_star=Path('t.star'), radius_step_a=50.0, walk_max_half_a=250.0, z_shift_a=60.0)

# _volume: noise with a dark ring of radius 84 voxels at (y, x) = (170, 170) present only in z = 25 to 55
def _volume() -> np.ndarray:
    rng = np.random.default_rng(1)
    vol = rng.normal(0.0, 1.0, (80, 340, 340)).astype(np.float32)
    yy, xx = np.mgrid[:340, :340]
    ring = np.exp(-0.5 * ((np.hypot(yy - 170, xx - 170) - 84) / 5.6) ** 2)
    vol[25:56] -= 1.5 * ring
    return vol

# test_ring_snr: a dark ring scores high, flat noise scores near zero
def test_ring_snr() -> None:
    config = _config()
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[:340, :340]
    ring = np.exp(-0.5 * ((np.hypot(yy - 170, xx - 170) - 84) / 5.6) ** 2)
    noise = rng.normal(0.0, 1.0, (340, 340))
    valid = np.ones((340, 340), dtype=bool)
    assert ring_snr(noise - 4.0 * ring, valid, (170.0, 170.0), 84.0, config, PX) > 3.0
    assert abs(ring_snr(noise, valid, (170.0, 170.0), 84.0, config, PX)) < 1.0

# test_estimate_candidate: an off-centre detection is pulled onto the ring, with a sensible radius, slab and box
def test_estimate_candidate() -> None:
    cand = Pick(tomogram='T', x=165, y=174, z=36, pixel_size_a=PX)
    est = estimate_candidate(_volume(), cand, _config())
    assert est.x == pytest.approx(170, abs=3)
    assert est.y == pytest.approx(170, abs=3)
    assert est.radius_a == pytest.approx(450.0, abs=30.0)
    assert est.z == pytest.approx(40, abs=8)
    assert est.z_lo <= 36 <= est.z_hi
    assert est.box_a == pytest.approx(2 * est.radius_a * 1.25)
    assert est.snr > 3.0
    assert 'edge' not in est.flag

# test_estimate_csv_round_trip: write then read keeps every field including an empty flag
def test_estimate_csv_round_trip(tmp_path: Path) -> None:
    est = Estimate(tomogram='T', x=1.0, y=2.0, z=3.0, pixel_size_a=PX, radius_a=450.0, box_a=1125.0, snr=4.2, z_lo=1, z_hi=5)
    write_estimates([est], tmp_path / 'e.csv')
    assert read_estimates(tmp_path / 'e.csv') == [est]
