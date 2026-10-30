'''
CaVe: tomogram discovery from tomograms.star, MRC loading and coordinate conversions
'''

# Import external dependencies
import argparse, csv, mrcfile, sys, numpy as np
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.text import Text
from typing import Literal

# Import internal CaVe objects
from cave.config import Config, load_config

# Tomogram: one reconstruction paired with its RELION star row; shape is (z, y, x) at the reconstruction sampling
@dataclass(frozen=True)
class Tomogram:
    # RELION name from tomograms.star, e.g. Position_10_4
    name: str
    # file stem of the reconstruction as RELION knows it, e.g. Position_10_4_stack_Vol
    stem: str
    raw_path: Path
    denoised_path: Path | None
    # reconstruction (binned) voxel size from the MRC header
    pixel_size: float
    shape: tuple[int, int, int]
    # tilt-series (unbinned) voxel size, and tomogram size in unbinned voxels (x, y, z), from the star file
    unbinned_pixel_size: float
    size_xyz: tuple[int, int, int]

# read_star_loop: rows of the first loop_ in a star file as column-name to string dicts
def read_star_loop(path: Path) -> list[dict[str, str]]:
    columns: list[str] = []
    rows: list[dict[str, str]] = []
    in_loop = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line == 'loop_':
            # a second loop means the first one is finished
            if in_loop:
                break
            in_loop = True
        elif line.startswith('data_'):
            if in_loop:
                break
        elif in_loop and line.startswith('_'):
            columns.append(line.split()[0].removeprefix('_'))
        elif in_loop:
            rows.append(dict(zip(columns, line.split(), strict=True)))
    return rows

# TomogramError: a tomogram that cannot be used; reason is the short label, str() the full message, the rest what was readable for the info table
class TomogramError(ValueError):
    def __init__(
        self, reason: str, detail: str, *,
        shape: tuple[int, int, int] | None = None, pixel_size: float | None = None, denoised: bool | None = None,
    ) -> None:
        super().__init__(detail)
        self.reason = reason
        self.shape = shape
        self.pixel_size = pixel_size
        self.denoised = denoised

# find_tomograms: pair every tomograms.star row with its raw and denoised MRC, keyed by RELION name; bad tomograms are skipped; problems go to the issues dict (name to reason) if given, else a warning on stderr
def find_tomograms(config: Config, *, issues: dict[str, TomogramError] | None = None) -> dict[str, Tomogram]:
    out: dict[str, Tomogram] = {}
    for row in read_star_loop(config.tomograms_star):
        name = row['rlnTomoName']
        try:
            out[name] = _build(config, row)
        except TomogramError as e:
            if issues is None:
                print(f'skipped: {name!r}: {e}', file=sys.stderr)
            else:
                issues[name] = e
    return out

# _build: the Tomogram for one star row, raising if files are missing or disagree with the star
def _build(config: Config, row: dict[str, str]) -> Tomogram:
    name = row['rlnTomoName']
    stem = Path(row['rlnTomoReconstructedTomogram']).stem
    raw = _match(config.raw_dir, name, stem, '')
    if raw is None:
        raise TomogramError('no raw tomogram', f'no raw tomogram in {str(config.raw_dir)!r}')
    denoised = _match(config.denoised_dir, name, stem, config.denoised_suffix) if config.denoised_dir else None
    pixel_size, shape = _header(raw)
    seen = {'shape': shape, 'pixel_size': pixel_size, 'denoised': denoised is not None}
    binning = float(row['rlnTomoTomogramBinning'])
    unbinned = float(row['rlnTomoTiltSeriesPixelSize'])
    size_xyz = (int(row['rlnTomoSizeX']), int(row['rlnTomoSizeY']), int(row['rlnTomoSizeZ']))
    # the header and the star file must describe the same reconstruction
    expected = tuple(round(n / binning) for n in size_xyz)
    if any(abs(a - b) > 1 for a, b in zip(expected, shape[::-1], strict=True)):
        raise TomogramError('shape mismatch', f'header shape (z, y, x) {shape!r} disagrees with star size {size_xyz!r} at binning {binning}', **seen)
    if abs(pixel_size / (unbinned * binning) - 1.0) > 0.02:
        raise TomogramError('pixel size mismatch', f'header pixel size {pixel_size:.2f} disagrees with {unbinned} x {binning}', **seen)
    return Tomogram(name, stem, raw, denoised, pixel_size, shape, unbinned, size_xyz)

# load_volume: read-only memory-mapped (z, y, x) volume; the map lives as long as the returned array
def load_volume(tomo: Tomogram, *, source: Literal['raw', 'denoised'] = 'denoised') -> np.ndarray:
    path = tomo.denoised_path if source == 'denoised' and tomo.denoised_path else tomo.raw_path
    return mrcfile.mmap(path, mode='r', permissive=True).data

# to_centred_angstrom: RELION centred (x, y, z) in Angstrom from (z, y, x) reconstruction voxels
def to_centred_angstrom(zyx: tuple[float, float, float], tomo: Tomogram) -> tuple[float, float, float]:
    z, y, x = zyx
    cx, cy, cz = (n * tomo.unbinned_pixel_size / 2 for n in tomo.size_xyz)
    return x * tomo.pixel_size - cx, y * tomo.pixel_size - cy, z * tomo.pixel_size - cz

# from_centred_angstrom: inverse of to_centred_angstrom, back to (z, y, x) reconstruction voxels
def from_centred_angstrom(xyz: tuple[float, float, float], tomo: Tomogram) -> tuple[float, float, float]:
    x, y, z = xyz
    cx, cy, cz = (n * tomo.unbinned_pixel_size / 2 for n in tomo.size_xyz)
    return (z + cz) / tomo.pixel_size, (y + cy) / tomo.pixel_size, (x + cx) / tomo.pixel_size

# run_info: summarise discovered tomograms and rejections; every tomogram's details with show_all
def run_info(config_path: Path, *, show_all: bool = False) -> None:
    config = load_config(config_path)
    issues: dict[str, TomogramError] = {}
    tomos = find_tomograms(config, issues=issues)
    console = Console()
    if not show_all:
        summary = Table(title='Summary')
        summary.add_column('Status')
        summary.add_column('Count', justify='right')
        summary.add_row('Usable', str(len(tomos)))
        summary.add_row('  with denoised pair', str(sum(t.denoised_path is not None for t in tomos.values())))
        summary.add_row('Rejected', str(len(issues)), style='red' if issues else None)
        console.print(summary)
        if issues:
            rejected = Table(title='Rejected tomograms')
            rejected.add_column('Name', overflow='fold')
            rejected.add_column('Reason', style='red')
            for name, e in issues.items():
                rejected.add_row(name, e.reason)
            console.print(rejected)
        return
    table = Table(title=f'{len(tomos) + len(issues)} tomogram(s)', caption=f'{len(tomos)} accepted, {len(issues)} rejected')
    table.add_column('Name', overflow='fold')
    for col in ('Shape (zyx)', 'Pixel size (A)', 'Star size (xyz)', 'Denoised'):
        table.add_column(col)
    table.add_column('Accepted', overflow='fold')
    # star order; the star size is always known, the rest only for tomograms whose files were readable
    for row in read_star_loop(config.tomograms_star):
        name = row['rlnTomoName']
        star_size = (int(row['rlnTomoSizeX']), int(row['rlnTomoSizeY']), int(row['rlnTomoSizeZ']))
        if name in tomos:
            t = tomos[name]
            shape, px, denoised, verdict = t.shape, t.pixel_size, t.denoised_path is not None, Text('\u2713', style='green')
        else:
            e = issues[name]
            shape, px, denoised, verdict = e.shape, e.pixel_size, e.denoised, Text.assemble(('\u2717', 'red'), f' ({e.reason})')
        table.add_row(
            name, '-' if shape is None else str(shape), '-' if px is None else f'{px:.2f}', str(star_size),
            '-' if denoised is None else 'yes' if denoised else 'no', verdict,
        )
    console.print(table)

# _stem_matches: whether a file stem is <run prefix> + reconstruction stem, the prefix being the one the merged name carries
def _stem_matches(stem: str, name: str, recon: str) -> bool:
    prefix = stem.removesuffix(recon)
    if prefix == stem and recon:
        return False
    # the name is <prefix><original RELION name>, and the original name begins the reconstruction stem
    return name.startswith(prefix) and recon.startswith(name.removeprefix(prefix))

# _match: the single MRC in a directory whose stem (minus suffix) matches the tomogram, or None
def _match(directory: Path, name: str, recon: str, suffix: str) -> Path | None:
    stems = [(p, p.stem.removesuffix(suffix)) for p in sorted(directory.glob('*.mrc'))]
    hits = [p for p, s in stems if _stem_matches(s, name, recon)]
    # unmerged star names carry no run prefix, so any prefixed file is accepted
    hits = hits or [p for p, s in stems if s == recon or s.endswith('_' + recon)]
    if len(hits) > 1:
        raise TomogramError('ambiguous match', f'{len(hits)} files in {str(directory)!r} match')
    return hits[0] if hits else None

# _header: voxel size and (z, y, x) shape from an MRC header without reading the data
def _header(path: Path) -> tuple[float, tuple[int, int, int]]:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        return float(mrc.voxel_size.x), (int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx))
