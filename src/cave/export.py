'''
CaVe: RELION-5 coordinate star files and a global extraction box recommendation from estimates
'''

# Import external dependencies
import numpy as np
from pathlib import Path

# Import internal CaVe objects
from cave.config import load_config
from cave.estimate import Estimate, read_estimates
from cave.io import Tomogram, find_tomograms, to_centred_angstrom

# write_particles_star: one data_particles table with RELION centred Angstrom coordinates
def write_particles_star(ests: list[Estimate], tomos: dict[str, Tomogram], path: Path) -> None:
    lines = [
        'data_particles', '', 'loop_',
        '_rlnTomoName #1', '_rlnCenteredCoordinateXAngst #2', '_rlnCenteredCoordinateYAngst #3', '_rlnCenteredCoordinateZAngst #4',
        '_rlnAutopickFigureOfMerit #5',
    ]
    for e in ests:
        x, y, z = to_centred_angstrom((e.z, e.y, e.x), tomos[e.tomogram])
        lines.append(f'{e.tomogram} {x:.2f} {y:.2f} {z:.2f} {e.snr:.3f}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')

# recommend_box: (box edge in Angstrom, box edge in pixels at extract_angpix rounded up to even) covering the given percentile of particles
def recommend_box(ests: list[Estimate], *, percentile: float, extract_angpix: float) -> tuple[float, int]:
    box_a = float(np.percentile([e.box_a for e in ests], percentile))
    box_px = int(np.ceil(box_a / extract_angpix))
    return box_a, box_px + box_px % 2

# run_export: write out/relion/particles.star and NAME_coords.star per tomogram, then print the box recommendation
def run_export(config_path: Path, min_snr: float, drop_flagged: bool, extract_angpix: float, percentile: float) -> None:
    config = load_config(config_path)
    tomos = find_tomograms(config)
    ests = [e for e in read_estimates(config.out_dir / 'estimates.csv') if e.snr >= min_snr and not (drop_flagged and e.flag)]
    if not ests:
        raise ValueError('no estimates left after filtering')
    out = config.out_dir / 'relion'
    write_particles_star(ests, tomos, out / 'particles.star')
    for name in tomos:
        mine = [e for e in ests if e.tomogram == name]
        if mine:
            write_particles_star(mine, tomos, out / f'{name}_coords.star')
    diameters = np.array([2 * e.radius_a for e in ests])
    box_a, box_px = recommend_box(ests, percentile=percentile, extract_angpix=extract_angpix)
    print(f'{len(ests)} particles, {sum(bool(e.flag) for e in ests)} flagged')
    print(f'diameter A: min {diameters.min():.0f}  median {np.median(diameters):.0f}  max {diameters.max():.0f}')
    print(f'recommended box: {box_a:.0f} A = {box_px} px at {extract_angpix} A/px (percentile {percentile:.0f})')
    print(f'wrote {out}')
