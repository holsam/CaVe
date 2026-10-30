'''
CaVe: command line interface and commands
'''

# Import external dependencies
import typer
from pathlib import Path
from typing import Annotated

# Import internal CaVe objects
from cave.estimate import run_estimate
from cave.export import run_export
from cave.extract import run_extract
from cave.io import run_info
from cave.merge import run_config
from cave.pick import run_pick

# Initialise Typer class for CLI
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
)

@app.command()
def info(
    config: Annotated[
        Path,
        typer.Argument(help='Path to config.toml (default: config.toml in the current directory).')
    ] = Path('config.toml'),
    show_all: Annotated[
        bool,
        typer.Option('--all', help='List every tomogram instead of a summary.')
    ] = False,
) -> None:
    '''Summarise identified tomograms and rejections.'''
    run_info(config, show_all=show_all)

@app.command()
def estimate(
    config: Annotated[
        Path,
        typer.Argument(help='Path to config.toml (default: config.toml in the current directory).')
    ] = Path('config.toml'),
) -> None:
    '''Refine every pick in out/picks.csv into a centre, radius and box and write out/estimates.csv.'''
    run_estimate(config)

@app.command()
def pick(
    config: Annotated[
        Path,
        typer.Argument(help='Path to config.toml (default: config.toml in the current directory).')
    ] = Path('config.toml'),
    tomos: Annotated[
        list[str] | None,
        typer.Option('-t', '--tomo', help='Name of tomogram to pick in (omit for all).')
    ] = None,
) -> None:
    '''Hand-pick caged vesicle centres in napari, saved to out/picks.csv.'''
    run_pick(config, tomos)

@app.command()
def extract(
    config: Annotated[
        Path,
        typer.Argument(help='Path to config.toml (default: config.toml in the current directory).')
    ] = Path('config.toml'),
    box_a: Annotated[
        float | None,
        typer.Option('-b', '--box-a', help='Box edge in Angstrom for every particle (default: the largest estimated box).')
    ] = None,
) -> None:
    '''Cut MRC boxes at the estimated centres in out/estimates.csv into out/particles.'''
    run_extract(config, box_a)

@app.command()
def config(
    projects: Annotated[
        list[str],
        typer.Argument(help='RUN=PROJECT_ROOT for each RELION project, e.g. raw2=/dls/.../raw2/relion_murfey.')
    ],
    raw_dir: Annotated[
        Path,
        typer.Option('--raw-dir', help='Directory of raw reconstructions.')
    ],
    denoised_dir: Annotated[
        Path | None,
        typer.Option('--denoised-dir', help='Directory of denoised reconstructions (omit for raw only).')
    ] = None,
    denoised_suffix: Annotated[
        str,
        typer.Option('--denoised-suffix', help='Suffix stripped from denoised stems to pair them with raw.')
    ] = '',
    tomo_job: Annotated[
        str,
        typer.Option('--tomo-job', help='Reconstruct job directory, relative to each project root.')
    ] = 'Tomograms/job006',
    outdir: Annotated[
        Path,
        typer.Option('-o', '--outdir', help='Output directory for the merged tomograms.star and tilt_series/ (default: current directory).')
    ] = Path('.'),
    config_path: Annotated[
        Path,
        typer.Option('-c', '--config', help='Where to write config.toml (must not exist).')
    ] = Path('config.toml'),
) -> None:
    '''Merge tomograms.star files from several RELION projects and write a config.toml for them.'''
    run_config(projects, tomo_job, outdir, config_path, raw_dir, denoised_dir, denoised_suffix)

@app.command()
def export(
    config: Annotated[
        Path,
        typer.Argument(help='Path to config.toml (default: config.toml in the current directory).')
    ] = Path('config.toml'),
    min_snr: Annotated[
        float,
        typer.Option('--min-snr', help='Drop estimates below this SNR.')
    ] = 0.0,
    drop_flagged: Annotated[
        bool,
        typer.Option('--drop-flagged', help='Drop estimates with any flag.')
    ] = False,
    extract_angpix: Annotated[
        float,
        typer.Option('--extract-angpix', help='Pixel size in Angstrom of the RELION extraction (bin 8 of 1.34 A by default).')
    ] = 10.72,
    percentile: Annotated[
        float,
        typer.Option('--percentile', help='Percentile of per-particle box width used for the recommended box.')
    ] = 95.0,
) -> None:
    '''Write RELION-5 coordinate star files to out/relion and recommend one global extraction box.'''
    run_export(config, min_snr, drop_flagged, extract_angpix, percentile)
