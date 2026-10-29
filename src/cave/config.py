'''
CaVe: pipeline configuration schema and TOML loader
'''

# Import external dependencies
import tomllib
from pathlib import Path
from pydantic import BaseModel
from typing import Literal

# Config: validated pipeline settings
class Config(BaseModel):
    # tomogram locations; denoised_dir None means raw only
    raw_dir: Path
    denoised_dir: Path | None = None
    # stripped from denoised stems so they pair with raw stems
    denoised_suffix: str = ''
    # RELION tomograms star: authoritative for names, unbinned pixel size, binning and tomogram size
    tomograms_star: Path
    # every CaVe output lands here
    out_dir: Path = Path('out')
    # volume the estimator and extractor read; falls back to raw when a tomogram has no denoised pair
    source: Literal['raw', 'denoised'] = 'denoised'
    # cage radius sweep and ring width, all in Angstrom
    cage_radius_range_a: tuple[float, float] = (150, 500.0)
    radius_step_a: float = 25.0
    ring_sigma_a: float = 30.0
    # background flattening and light smoothing of each projection
    highpass_sigma_a: float = 500.0
    smooth_sigma_a: float = 8.0
    # estimation: ring half-width and the background annulus as multiples of the ring radius
    ring_halfwidth_a: float = 40.0
    bg_range: tuple[float, float] = (1.3, 1.8)
    # estimation: z slab grown up to walk_max_half_a, stopping after walk_patience steps without a better SNR
    walk_max_half_a: float = 400.0
    walk_patience: int = 3
    # estimation: how far the centre may move from the pick in xy, and how far the slab centre may shift in z
    locate_range_a: float = 200.0
    z_shift_a: float = 100.0
    # extraction box edge as a multiple of the cage diameter
    box_margin: float = 1.25

# load_config: read a TOML file into a Config, resolving relative paths against the file
def load_config(path: Path) -> Config:
    if not path.is_file():
        raise FileNotFoundError(f'{path!r} not found, enter the path to config.toml as an argument')
    with open(path, 'rb') as handle:
        config = Config(**tomllib.load(handle))
    base = path.resolve().parent
    for field in ('raw_dir', 'denoised_dir', 'tomograms_star', 'out_dir'):
        value = getattr(config, field)
        if value is not None and not value.is_absolute():
            setattr(config, field, base / value)
    return config
