'''
CaVe: tomogram discovery from tomograms.star, MRC loading and coordinate conversions
'''

# Import external dependencies
import argparse, csv, mrcfile, numpy as np
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
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

# find_tomograms: pair every tomograms.star row with its raw and denoised MRC, keyed by RELION name
def find_tomograms(config: Config) -> dict[str, Tomogram]:
    out: dict[str, Tomogram] = {}
    for row in read_star_loop(config.tomograms_star):
        name = row['rlnTomoName']
        stem = Path(row['rlnTomoReconstructedTomogram']).stem
        raw = _match(config.raw_dir, name, stem, '')
        if raw is None:
            raise FileNotFoundError(f'no raw tomogram for {name!r} in {config.raw_dir!r}')
        denoised = _match(config.denoised_dir, name, stem, config.denoised_suffix) if config.denoised_dir else None
        pixel_size, shape = _header(raw)
        binning = float(row['rlnTomoTomogramBinning'])
        unbinned = float(row['rlnTomoTiltSeriesPixelSize'])
        size_xyz = (int(row['rlnTomoSizeX']), int(row['rlnTomoSizeY']), int(row['rlnTomoSizeZ']))
        # the header and the star file must describe the same reconstruction
        expected = tuple(round(n / binning) for n in size_xyz)
        if any(abs(a - b) > 1 for a, b in zip(expected, shape[::-1], strict=True)):
            raise ValueError(f'{name!r}: header shape (z, y, x) {shape!r} disagrees with star size {size_xyz!r} at binning {binning}')
        if abs(pixel_size / (unbinned * binning) - 1.0) > 0.02:
            raise ValueError(f'{name!r}: header pixel size {pixel_size} disagrees with {unbinned} x {binning}')
        out[name] = Tomogram(name, stem, raw, denoised, pixel_size, shape, unbinned, size_xyz)
    return out

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

# run_info: list tomograms and their sampling
def run_info(config_path: Path) -> None:
    config = load_config(config_path)
    tomos = find_tomograms(config)
    for t in tomos.values():
        print(
            f'{t.name}  shape(zyx)={t.shape}  {t.pixel_size:.2f} A  star size(xyz)={t.size_xyz}'
            f'  denoised={t.denoised_path is not None}'
        )

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
        raise ValueError(f'{len(hits)} files in {directory!r} match {name!r}')
    return hits[0] if hits else None

# _header: voxel size and (z, y, x) shape from an MRC header without reading the data
def _header(path: Path) -> tuple[float, tuple[int, int, int]]:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        return float(mrc.voxel_size.x), (int(mrc.header.nz), int(mrc.header.ny), int(mrc.header.nx))
