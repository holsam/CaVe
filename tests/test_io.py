'''
CaVe: IO tests using a synthetic tomogram and star file.
'''

# Import external dependencies
import mrcfile, numpy as np, pytest
from pathlib import Path

# Import internal CaVe objects
from cave.config import Config
from cave.io import find_tomograms, from_centred_angstrom, load_volume, read_star_loop, to_centred_angstrom

# STAR: one tomogram, 8 x 6 x 4 voxels at 5.36 A (bin 4 of 1.34 A)
STAR = '''data_global

loop_
_rlnTomoName #1
_rlnTomoTiltSeriesPixelSize #2
_rlnTomoTomogramBinning #3
_rlnTomoSizeX #4
_rlnTomoSizeY #5
_rlnTomoSizeZ #6
_rlnTomoReconstructedTomogram #7
Position_1_2 1.34 4.0 32 24 16 Tomograms/job006/tomograms/Position_1_2_stack_Vol.mrc
'''

# _mrc: write a tiny MRC with the given voxel size and (z, y, x) shape
def _mrc(path: Path, voxel: float, shape: tuple[int, int, int]) -> None:
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.arange(np.prod(shape), dtype=np.float32).reshape(shape))
        mrc.voxel_size = voxel

# _config: a project on disk with raw, denoised and star
def _config(tmp_path: Path, shape: tuple[int, int, int] = (4, 6, 8)) -> Config:
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'den').mkdir()
    _mrc(tmp_path / 'raw' / 'nt_raw4_Position_1_2_stack_Vol.mrc', 5.36, shape)
    _mrc(tmp_path / 'den' / 'nt_raw4_Position_1_2_stack_Vol.denoised.mrc', 5.36, shape)
    (tmp_path / 't.star').write_text(STAR)
    return Config(raw_dir=tmp_path / 'raw', denoised_dir=tmp_path / 'den', denoised_suffix='.denoised', tomograms_star=tmp_path / 't.star')

# test_star_loop: columns and rows of the first loop
def test_star_loop(tmp_path: Path) -> None:
    (tmp_path / 't.star').write_text(STAR)
    rows = read_star_loop(tmp_path / 't.star')
    assert rows[0]['rlnTomoName'] == 'Position_1_2'
    assert rows[0]['rlnTomoSizeZ'] == '16'

# test_find_tomograms: pairing by stem, header values and star values
def test_find_tomograms(tmp_path: Path) -> None:
    tomos = find_tomograms(_config(tmp_path))
    t = tomos['Position_1_2']
    assert t.shape == (4, 6, 8)
    assert t.pixel_size == pytest.approx(5.36, rel=1e-4)
    assert t.size_xyz == (32, 24, 16)
    assert t.denoised_path is not None
    assert load_volume(t, source='denoised').shape == (4, 6, 8)

# test_find_tomograms_shape_mismatch: a header that disagrees with the star file is skipped and reported as an issue
def test_find_tomograms_shape_mismatch(tmp_path: Path) -> None:
    issues: dict[str, str] = {}
    tomos = find_tomograms(_config(tmp_path, shape=(4, 6, 20)), issues=issues)
    assert tomos == {}
    assert len(issues) == 1 and 'disagrees' in next(iter(issues.values()))

# test_centred_round_trip: the tomogram centre is the origin and the conversion inverts
def test_centred_round_trip(tmp_path: Path) -> None:
    t = find_tomograms(_config(tmp_path))['Position_1_2']
    assert to_centred_angstrom((2.0, 3.0, 4.0), t) == pytest.approx((0.0, 0.0, 0.0), abs=1e-2)
    zyx = (1.5, 2.5, 6.0)
    assert from_centred_angstrom(to_centred_angstrom(zyx, t), t) == pytest.approx(zyx)