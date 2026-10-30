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

# Progress: where picking stopped and which tomograms have been opened, saved beside picks.csv
class Progress(BaseModel):
    last: str | None = None
    visited: list[str] = []

# write_progress: dump progress to JSON
def write_progress(progress: Progress, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(progress.model_dump_json())

# read_progress: load progress, empty if the file does not exist yet
def read_progress(path: Path) -> Progress:
    return Progress.model_validate_json(path.read_text()) if path.exists() else Progress()

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
        # resume at the tomogram shown last time, if it is among those being picked
        self.progress_path = csv_path.with_name('progress.json')
        saved = read_progress(self.progress_path)
        self.visited = set(saved.visited)
        names = [t.name for t in tomos]
        self.index = 0
        self._loading = False
        self._marker_px = 0.0
        self.go(names.index(saved.last) if saved.last in names else 0)

    # go: show tomogram i at its middle Z with its saved picks; clamped, so no wrapping
    def go(self, i: int) -> None:
        self.index = min(max(i, 0), len(self.tomos) - 1)
        tomo = self.tomos[self.index]
        self._loading = True
        try:
            # a selection from the previous tomogram would otherwise index the new, shorter features
            self.points.selected_data = set()
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
        self.visited.add(tomo.name)
        write_progress(Progress(last=tomo.name, visited=sorted(self.visited)), self.progress_path)
        self.refresh()

    # step: move by delta tomograms
    def step(self, delta: int) -> None:
        self.go(self.index + delta)

    # jump: go to a tomogram by 1-based number or by name (exact match first, else the first containing the text); False if nothing matches
    def jump(self, query: str) -> bool:
        query = query.strip().lower()
        names = [t.name.lower() for t in self.tomos]
        if query.isdigit():
            hit = int(query) - 1 if 0 < int(query) <= len(names) else None
        elif query:
            hit = names.index(query) if query in names else next((i for i, n in enumerate(names) if query in n), None)
        else:
            hit = None
        if hit is None:
            return False
        self.go(hit)
        return True

    # set_z: the z that newly clicked points are given, kept in step with the viewer's slice
    def set_z(self, z: float) -> None:
        self.points.feature_defaults = {'z': float(z)}

    # show_selected: jump to the z of a single selected pick, so clicking a finished pick lands on its slice
    def show_selected(self) -> None:
        selected = list(self.points.selected_data)
        # napari fires highlight mid-swap, while the features still belong to the old tomogram
        if not self._loading and len(selected) == 1 and selected[0] < len(self.points.features):
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

    # summary: one line per tomogram with its pick count, the current one marked > and opened ones *, then the visited count and the total
    def summary(self) -> str:
        lines = [f'{">" if i == self.index else " "} {"*" if t.name in self.visited else " "} {t.name}  {len(self.picks[t.name])}' for i, t in enumerate(self.tomos)]
        opened = sum(t.name in self.visited for t in self.tomos)
        return '\n'.join(lines + [f'  visited  {opened}/{len(self.tomos)}', f'  total  {sum(map(len, self.picks.values()))}'])

    # refresh: update the counts box and the overlay
    def refresh(self) -> None:
        text = self.summary()
        if self.label is not None:
            self.label.setText(text)
        self.viewer.text_overlay.text = f'{self.tomos[self.index].name}  ({self.index + 1}/{len(self.tomos)})  Enter next  Shift+Enter back'
        self.viewer.text_overlay.visible = True

# install_controls: window-level Enter and Shift+Enter shortcuts, independent of canvas keyboard focus, a jump box and the scrollable counts box
def install_controls(window, picker: Picker):
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFontDatabase, QKeySequence
    from qtpy.QtWidgets import QLabel, QLineEdit, QScrollArea, QShortcut, QVBoxLayout, QWidget

    box = QLabel()
    box.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    picker.label = box
    picker.refresh()
    jump = QLineEdit()
    jump.setPlaceholderText('go to number or name, then Enter')

    # Enter inside the jump box goes to the typed tomogram, anywhere else it steps on
    def on_enter(delta: int) -> None:
        if not jump.hasFocus():
            picker.step(delta)
        elif delta > 0 and picker.jump(jump.text()):
            jump.clear()
            jump.clearFocus()

    # the main and keypad Enter keys both count
    shortcuts = []
    for keys, delta in (('Return', 1), ('Enter', 1), ('Shift+Return', -1), ('Shift+Enter', -1)):
        shortcut = QShortcut(QKeySequence(keys), window)
        shortcut.setContext(Qt.WindowShortcut)
        shortcut.activated.connect(lambda delta=delta: on_enter(delta))
        shortcuts.append(shortcut)
    # keep the shortcuts alive with the box
    box._shortcuts = shortcuts
    # one line per tomogram would otherwise force the window as tall as the list, and a canvas over the GL texture limit cannot draw
    area = QScrollArea()
    area.setWidget(box)
    area.setWidgetResizable(True)
    area.setMinimumHeight(100)
    area._box = box
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(jump)
    layout.addWidget(area)
    return panel

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
