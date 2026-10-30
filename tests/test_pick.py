'''
CaVe: pick state, autosave, resume and navigation, with a stand-in for the napari viewer
'''

# Import external dependencies
import mrcfile, numpy as np, pandas as pd
from pathlib import Path
from types import SimpleNamespace

# Import internal CaVe objects
from cave.config import Config
from cave.io import Tomogram
from cave.pick import Picker, read_picks

# _viewer: just the attributes Picker drives
def _viewer() -> SimpleNamespace:
    return SimpleNamespace(
        dims=SimpleNamespace(set_point=lambda axis, value: setattr(_viewer, 'z', value)),
        reset_view=lambda: None,
        text_overlay=SimpleNamespace(text='', visible=False),
    )

# _setup: two tiny tomograms on disk, and the layers a Picker drives
def _setup(tmp_path: Path) -> tuple[list[Tomogram], Config, SimpleNamespace, SimpleNamespace]:
    tomos = []
    for name in ('A', 'B'):
        path = tmp_path / f'{name}.mrc'
        with mrcfile.new(path) as mrc:
            mrc.set_data(np.zeros((10, 8, 8), dtype=np.float32))
        tomos.append(Tomogram(name, name, path, None, 5.36, (10, 8, 8), 1.34, (32, 32, 40)))
    return tomos, Config(raw_dir=tmp_path, tomograms_star=tmp_path / 't.star', source='raw'), SimpleNamespace(data=None, contrast_limits=None), SimpleNamespace(data=None, size=None, features=None, feature_defaults=None, current_size=None, selected_data=set())

# test_pick_save_and_resume: picks autosave per tomogram, navigation keeps them apart, and a new Picker restores them
def test_pick_save_and_resume(tmp_path: Path) -> None:
    tomos, config, image, points = _setup(tmp_path)
    csv_path = tmp_path / 'picks.csv'
    picker = Picker(_viewer(), tomos, config, csv_path, image, points)
    assert _viewer.z == 5
    points.data, points.features = np.array([[2.0, 3.0], [4.0, 1.0]]), pd.DataFrame({'z': [5.0, 5.0]})
    picker.sync()
    picker.step(1)
    assert len(points.data) == 0
    points.data, points.features = np.array([[1.0, 1.0]]), pd.DataFrame({'z': [4.0]})
    picker.sync()
    assert points.feature_defaults == {'z': 5.0}
    assert [(p.tomogram, p.x, p.y, p.z) for p in read_picks(csv_path)] == [('A', 3.0, 2.0, 5.0), ('A', 1.0, 4.0, 5.0), ('B', 1.0, 1.0, 4.0)]
    assert picker.summary().splitlines()[-1].endswith('3')
    # selecting one pick jumps to its z
    picker.step(-1)
    _viewer.z = 0
    points.selected_data = {1}
    picker.show_selected()
    assert _viewer.z == 5.0
    # a stale selection must not break navigation, and is cleared on the way
    points.selected_data = {1}
    picker.step(1)
    assert points.selected_data == set()
    picker.step(-1)
    # a selection beyond the features is ignored, not an IndexError
    points.selected_data = {99}
    picker.show_selected()
    picker.step(1)
    # clamped at both ends
    picker.step(1)
    assert picker.index == 1
    picker.step(-5)
    assert picker.index == 0
    resumed = Picker(_viewer(), tomos, config, csv_path, image, points)
    assert points.data.tolist() == [[2.0, 3.0], [4.0, 1.0]] and points.features['z'].tolist() == [5.0, 5.0]
    assert resumed.picks['B'] == [(4.0, 1.0, 1.0)]

# test_progress_and_jump: the last tomogram is reopened on the next run, opened tomograms are marked, and jump takes a number or a name
def test_progress_and_jump(tmp_path: Path) -> None:
    tomos, config, image, points = _setup(tmp_path)
    csv_path = tmp_path / 'picks.csv'
    picker = Picker(_viewer(), tomos, config, csv_path, image, points)
    assert picker.index == 0 and picker.summary().splitlines()[-2].endswith('1/2')
    assert picker.jump('2') and picker.index == 1
    assert picker.jump('a') and picker.index == 0
    assert not picker.jump('') and not picker.jump('9') and not picker.jump('zzz')
    picker.step(1)
    assert Picker(_viewer(), tomos, config, csv_path, image, points).index == 1
    assert [line[2] for line in picker.summary().splitlines()[:2]] == ['*', '*']
