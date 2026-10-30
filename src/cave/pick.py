'''
CaVe: napari manual picking of one caged vesicle centre per click, autosaved to picks.csv
'''

# Import external dependencies
import csv, numpy as np, pandas as pd
from pathlib import Path
from pydantic import BaseModel

# Import internal CaVe objects
from cave.config import Config, load_config
from cave.io import Tomogram, find_tomograms, load_volume

# _MARKER_A: display diameter of a pick marker in Angstrom, cosmetic only
_MARKER_A: float = 600.0

# Pick: one hand-picked centre; x, y, z in reconstruction voxels like Candidate
class Pick(BaseModel):
    tomogram: str
    x: float
    y: float
    z: float
    pixel_size_a: float

# write_picks: dump picks to CSV
def write_picks(picks: list[Pick], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Pick.model_fields))
        writer.writeheader()
        writer.writerows(p.model_dump() for p in picks)

# read_picks: load picks, empty if the file does not exist yet
def read_picks(path: Path) -> list[Pick]:
    if not path.exists():
        return []
    with open(path, newline='') as handle:
        return [Pick(**row) for row in csv.DictReader(handle)]

# Picker: pick state and tomogram navigation, driving any napari viewer with an image layer and a points layer
class Picker:
    # points is a 2D (y, x) layer, so every pick shows at every Z, with each pick's z kept as a feature; label is anything with setText, e.g. a QLabel, for the counts
    def __init__(self, viewer, tomos: list[Tomogram], config: Config, csv_path: Path, image, points, *, label=None) -> None:
        self.viewer, self.tomos, self.config, self.csv_path = viewer, tomos, config, csv_path
        self.image, self.points, self.label = image, points, label
        self.picks: dict[str, list[tuple[float, float, float]]] = {t.name: [] for t in tomos}
        for p in read_picks(csv_path):
            if p.tomogram in self.picks:
                self.picks[p.tomogram].append((p.z, p.y, p.x))
        self.index = 0
        self._loading = False
        self._marker_px = 0.0
        self.go(0)

    # go: show tomogram i at its middle Z with its saved picks; clamped, so no wrapping
    def go(self, i: int) -> None:
        self.index = min(max(i, 0), len(self.tomos) - 1)
        tomo = self.tomos[self.index]
        self._loading = True
        try:
            vol = load_volume(tomo, source=self.config.source)
            self.image.data = vol
            self.image.contrast_limits = tuple(float(v) for v in np.percentile(vol[::8, ::8, ::8], (1, 99)))
            zyx = np.array(self.picks[tomo.name], dtype=float).reshape(-1, 3)
            self.points.data = zyx[:, 1:]
            self.points.features = pd.DataFrame({'z': zyx[:, 0]})
            self._marker_px = _MARKER_A / tomo.pixel_size
            # current_size is what newly clicked points get; size alone only resizes existing ones
            self.points.current_size = self._marker_px
            self.points.size = self._marker_px
        finally:
            self._loading = False
        self.viewer.dims.set_point(0, tomo.shape[0] // 2)
        self.set_z(tomo.shape[0] // 2)
        self.viewer.reset_view()
        self.refresh()

    # step: move by delta tomograms
    def step(self, delta: int) -> None:
        self.go(self.index + delta)

    # set_z: the z that newly clicked points are given, kept in step with the viewer's slice
    def set_z(self, z: float) -> None:
        self.points.feature_defaults = {'z': float(z)}

    # show_selected: jump to the z of a single selected pick, so clicking a finished pick lands on its slice
    def show_selected(self) -> None:
        selected = list(self.points.selected_data)
        if len(selected) == 1:
            self.viewer.dims.set_point(0, float(self.points.features['z'].iloc[selected[0]]))

    # sync: take the points layer as this tomogram's picks and save; wired to the layer's data event
    def sync(self) -> None:
        if self._loading:
            return
        yx = np.asarray(self.points.data, dtype=float).reshape(-1, 2)
        z = np.asarray(self.points.features['z'], dtype=float)
        self.picks[self.tomos[self.index].name] = [(float(z[i]), float(y), float(x)) for i, (y, x) in enumerate(yx)]
        self.save()
        self.refresh()

    # save: write every tomogram's picks
    def save(self) -> None:
        write_picks(
            [Pick(tomogram=t.name, x=x, y=y, z=z, pixel_size_a=t.pixel_size) for t in self.tomos for z, y, x in self.picks[t.name]],
            self.csv_path,
        )

    # summary: one line per tomogram with its pick count, the current one marked, and the total
    def summary(self) -> str:
        lines = [f'{">" if i == self.index else " "} {t.name}  {len(self.picks[t.name])}' for i, t in enumerate(self.tomos)]
        return '\n'.join(lines + [f'  total  {sum(map(len, self.picks.values()))}'])

    # refresh: update the counts box and the overlay
    def refresh(self) -> None:
        text = self.summary()
        if self.label is not None:
            self.label.setText(text)
        self.viewer.text_overlay.text = f'{self.tomos[self.index].name}  ({self.index + 1}/{len(self.tomos)})  Enter next  Shift+Enter back'
        self.viewer.text_overlay.visible = True

# install_controls: window-level Enter and Shift+Enter shortcuts, independent of canvas keyboard focus, and the scrollable counts box
def install_controls(window, picker: Picker):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFontDatabase, QKeySequence
    from qtpy.QtWidgets import QLabel, QScrollArea, QShortcut

    box = QLabel()
    box.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    picker.label = box
    picker.refresh()
    # the main and keypad Enter keys both count
    shortcuts = []
    for keys, delta in (('Return', 1), ('Enter', 1), ('Shift+Return', -1), ('Shift+Enter', -1)):
        shortcut = QShortcut(QKeySequence(keys), window)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(lambda delta=delta: picker.step(delta))
        shortcuts.append(shortcut)
    # keep the shortcuts alive with the box
    box._shortcuts = shortcuts
    # one line per tomogram would otherwise force the window as tall as the list, and a canvas over the GL texture limit cannot draw
    area = QScrollArea()
    area.setWidget(box)
    area.setWidgetResizable(True)
    area.setMinimumHeight(100)
    area._box = box
    return area

# run_pick: one napari session over every tomogram, opening each at middle Z in add mode
def run_pick(config_path: Path, tomos: list[str] | None) -> None:
    import napari

    config = load_config(config_path)
    selected = [t for name, t in find_tomograms(config).items() if tomos is None or name in tomos]
    viewer = napari.Viewer(ndisplay=2)
    image = viewer.add_image(np.zeros((1, 4, 4), dtype=np.float32), name='tomogram', colormap='gray')
    # 2D (y, x) layer shows at every Z; translucent yellow disc, z kept in the 'z' feature
    points = viewer.add_points(np.empty((0, 2)), name='picks', ndim=2, symbol='disc', face_color=[1, 1, 0, 0.3], border_color='yellow', features={'z': np.array([], dtype=float)})
    points.mode = 'add'
    picker = Picker(viewer, selected, config, config.out_dir / 'picks.csv', image, points)
    viewer.dims.events.current_step.connect(lambda _event: picker.set_z(viewer.dims.current_step[0]))
    points.events.highlight.connect(lambda _event: picker.show_selected())
    points.events.data.connect(lambda _event: picker.sync())
    viewer.window.add_dock_widget(install_controls(viewer.window._qt_window, picker), name='picks', area='left')
    viewer.text_overlay.position = 'top_left'
    viewer.text_overlay.font_size = 11
    viewer.text_overlay.color = 'yellow'
    napari.run()
