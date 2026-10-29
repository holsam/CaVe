'''
CaVe: cubic MRC boxes cut from the tomograms at the estimated cage centres
'''

# Import external dependencies
import mrcfile, numpy as np
from pathlib import Path

# Import internal CaVe objects
from cave.config import load_config
from cave.estimate import read_estimates
from cave.io import Tomogram, find_tomograms, load_volume, read_star_loop, to_centred_angstrom

# crop_box: cubic crop centred on (z, y, x) voxel indices, padded with the crop's own mean where it leaves the volume
def crop_box(vol: np.ndarray, centre: tuple[int, int, int], box: int) -> np.ndarray:
    half = box // 2
    src = tuple(slice(max(c - half, 0), min(c - half + box, n)) for c, n in zip(centre, vol.shape, strict=True))
    block = np.asarray(vol[src], dtype=np.float32)
    out = np.full((box, box, box), block.mean(), dtype=np.float32)
    out[tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(src, centre, strict=True))] = block
    return out

# run_extract: writes out/particles/NAME/NAME_NNN.mrc per estimate; one box for all, the largest estimate unless box_a is given
def run_extract(config_path: Path, box_a: float | None) -> None:
    config = load_config(config_path)
    ests = read_estimates(config.out_dir / 'estimates.csv')
    tomos = {name: t for name, t in find_tomograms(config).items() if any(e.tomogram == name for e in ests)}
    if len({t.pixel_size for t in tomos.values()}) > 1:
        raise ValueError(f'one box in pixels needs one pixel size, got {sorted({t.pixel_size for t in tomos.values()})!r}')
    px = next(iter(tomos.values())).pixel_size
    # even box, so the centre voxel sits at box // 2; RELION needs the same box for every particle
    box = int(round((box_a or max(e.box_a for e in ests)) / px / 2)) * 2
    for name, tomo in tomos.items():
        vol = load_volume(tomo, source=config.source)
        mine = [e for e in ests if e.tomogram == name]
        for i, e in enumerate(mine):
            dest = config.out_dir / 'particles' / name / f'{name}_{i:03d}.mrc'
            dest.parent.mkdir(parents=True, exist_ok=True)
            with mrcfile.new(dest, overwrite=True) as mrc:
                mrc.set_data(crop_box(vol, (round(e.z), round(e.y), round(e.x)), box))
                mrc.voxel_size = tomo.pixel_size
        print(f'{name}: {len(mine)} boxes')
    print(f'box {box} px ({box * px:.0f} A)')
