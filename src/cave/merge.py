'''
CaVe: merge tomograms.star files from several RELION projects into one with absolute paths
'''

# Import external dependencies
import json
from pathlib import Path

# Import internal CaVe objects
from cave.io import read_star_loop

# run_config: merge the projects' tomograms.star files, then write a config.toml pointing at the result
def run_config(projects: list[str], tomo_job: str, outdir: Path, config_path: Path, raw_dir: Path, denoised_dir: Path | None, denoised_suffix: str) -> None:
    for path in (config_path, outdir / 'tomograms.star'):
        if path.exists():
            raise FileExistsError(f'{path!r} already exists')
    star = run_merge(projects, tomo_job, outdir)
    lines = [f'raw_dir = {json.dumps(str(raw_dir.resolve()))}']
    if denoised_dir is not None:
        lines += [f'denoised_dir = {json.dumps(str(denoised_dir.resolve()))}', f'denoised_suffix = {json.dumps(denoised_suffix)}']
    lines += [f'tomograms_star = {json.dumps(str(star))}', 'out_dir = "out"']
    config_path.write_text('\n'.join(lines) + '\n')
    print(f'wrote {config_path}')

# run_merge: write a merged tomograms.star plus per-tomogram tilt-series stars, tomogram names prefixed by run
def run_merge(projects: list[str], tomo_job: str, outdir: Path) -> Path:
    outdir = outdir.resolve()  # star files must hold absolute paths
    (outdir / 'tilt_series').mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for spec in projects:
        run, sep, root_str = spec.partition('=')
        if not sep:
            raise ValueError(f'expected RUN=PROJECT_ROOT, got {spec!r}')
        root = Path(root_str)
        for row in read_star_loop(root / tomo_job / 'tomograms.star'):
            new = f'{run}_{row["rlnTomoName"]}'
            # first occurrence wins when a project lists a tomogram twice
            if new in seen:
                print(f'skipping duplicate tomogram {new!r}')
                continue
            seen.add(new)
            ts_rows = read_star_loop(root / row['rlnTomoTiltSeriesStarFile'])
            for r in ts_rows:
                r['rlnMicrographName'] = _absolute(root, r['rlnMicrographName'])
            ts_path = outdir / 'tilt_series' / f'{new}.star'
            _write_loop(new, ts_rows, ts_path)
            row['rlnTomoName'] = new
            row['rlnTomoTiltSeriesStarFile'] = str(ts_path)
            row['rlnTomoReconstructedTomogram'] = _absolute(root, row['rlnTomoReconstructedTomogram'])
            merged.append(row)
    if not merged:
        raise ValueError('no tomograms found in the given projects')
    _write_loop('global', merged, outdir / 'tomograms.star')
    print(f'wrote {len(merged)} tomograms to {outdir / "tomograms.star"}')
    return outdir / 'tomograms.star'

# _absolute: resolve a path relative to a project root, leaving absolute paths alone
def _absolute(root: Path, value: str) -> str:
    return value if value.startswith('/') else str(root / value)

# _write_loop: write rows as a single-loop star data block
def _write_loop(name: str, rows: list[dict[str, str]], path: Path) -> None:
    cols = list(rows[0]) if rows else []
    lines = [f'data_{name}', '', 'loop_'] + [f'_{c} #{i + 1}' for i, c in enumerate(cols)]
    path.write_text('\n'.join(lines + [' '.join(r[c] for c in cols) for r in rows]) + '\n')
