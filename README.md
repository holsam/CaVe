# CaVe

Caged vesicle picking, box estimation and extraction for STA.

## Install

```sh
uv tool install git+https://github.com/holsam/cave
cave --help
```

## Workflow

Every command except `config` reads `config.toml` from the current directory (or from a path given as its first argument) and writes into `out_dir` (default `out/`).

```sh
cave config raw2=/path/to/relion_a raw3=/path/to/relion_b --raw-dir DIR   # merge RELION projects, write config.toml
cave info                                                                 # check tomograms are found and paired
cave pick [-t NAME]                                                       # click cage centres in napari
cave estimate                                                             # refine centre, radius and box
cave extract [-b BOX_A]                                                   # cut MRC boxes (optional)
cave export                                                               # RELION coordinates and box advice
```

### `config`
Merges multiple `tomograms.star` from RELION projects into `tomograms.star` with absolute paths and prefixed tomogram names (`raw2_Position_1_2`), then creates the `config.toml` required for CaVe commands. Refuses to overwrite an existing `config.toml` or `tomograms.star`. Skip if you already have both.

Option | Description
-- | --
`--raw-dir` | Directory containing reconstructed tomograms (required)
`--denoised-dir` | Directory containing denoised tomograms
`--denoised-suffix` | Suffix stripped from denoised stems to pair them with raw
`--tomo-job` | Reconstruct job directory in each project (default `Tomograms/job006`)
`-o`, `--outdir` | Directory for `tomograms.star` and `tilt_series/` (default `.`)
`-c`, `--config` | Where to write `config.toml` (default `config.toml`)

### `info`
Summary table of usable and rejected tomograms (rejections listed with reasons). `--all` lists each tomogram with its shape, pixel size and whether a denoised pair exists.

### `pick`
Napari picking of cage centres, autosaved to `out/picks.csv`. Reruns resume from existing picks.

Option | Description
-- | --
`-t`, `--tomo` | Tomogram to pick in (repeatable; default all)

### `estimate`
Refines every pick into a centre, radius, box and SNR, written to `out/estimates.csv`. Estimates may carry flags (`edge`, `zedge`, `zmax`).

### `extract`
Cuts one uniform box per estimate into `out/particles/NAME/NAME_NNN.mrc`. Optional, not needed for the RELION handoff.

Option | Description
-- | --
`-b`, `--box-a` | Box edge in Angstrom (default largest estimate)

### `export`
Writes `out/relion/particles.star` (all tomograms) and `NAME_coords.star` per tomogram, in centred Angstrom with `rlnAutopickFigureOfMerit` set to the SNR. Prints the diameter range and a recommended global box.

Option | Description
-- | --
`--min-snr` | Drop estimates below this SNR (default 0)
`--drop-flagged` | Drop estimates with any flag
`--extract-angpix` | Pixel size of the RELION extraction (default 10.72, bin 8 of 1.34 A)
`--percentile` | Percentile of box width for the recommended box (default 95)

## RELION handoff

1. Import the tomograms with the same `tomograms.star` (or its optimisation set).
2. Run *Import coordinates* with `out/relion/particles.star` and the tomograms set. Check all particles are recognised; a name mismatch gives zero or few.
3. Reconstruct the particles at coarse binning with no alignment. Expect a centred dense vesicle in a roughly spherical cage. Mirror-image positions mean handedness or the z origin disagrees with `tomograms.star`.
4. Extract pseudo-subtomograms using the box recommended by `cave export`, binned to match `--extract-angpix`.
5. Make a first reference from the unaligned average, low-pass filtered hard (about 50 A), then continue with refinement and classification.

## Config

```toml
raw_dir = "./data/raw"                # raw reconstructions
denoised_dir = "./data/denoised"      # optional; omit for raw only
denoised_suffix = ".denoised"         # stripped from denoised stems to pair with raw
tomograms_star = "./tomograms.star"   # authoritative for names, pixel size and tomogram size
out_dir = "out"
source = "denoised"                   # volume read by estimate and extract: raw or denoised
highpass_sigma_a = 1000
```

Estimation settings (cage radius range, ring widths, z walk limits and so on) have defaults documented in `src/cave/config.py`.

## Tests

```sh
uv run pytest
```
