'''
CaVe: per-cage centre, radius and box width from a z slab grown until ring SNR stops improving
'''

# Import external dependencies
import argparse, csv, numpy as np
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel
from skimage.feature import match_template

# Import internal CaVe objects
from cave.config import Config, load_config
from cave.io import find_tomograms, load_volume
from cave.project import highpass, smooth
from cave.pick import Pick, read_picks

# Estimate: cage geometry from the best slab; x, y, z in reconstruction voxels, radius and box in Angstrom
class Estimate(BaseModel):
    tomogram: str
    x: float
    y: float
    z: float
    pixel_size_a: float
    radius_a: float
    box_a: float
    snr: float
    # slab of the best SNR as inclusive slice indices
    z_lo: int
    z_hi: int
    # comma-separated warnings: edge, zedge, zmax
    flag: str = ''

# Step: ring fit for one slab half-thickness k; y, x are crop coordinates, radius_px in voxels
@dataclass(frozen=True)
class Step:
    k: int
    y: float
    x: float
    radius_px: float
    snr: float

# ring_template: dark gaussian ring of the given radius, zero elsewhere (low where the image should be low)
def ring_template(radius_px: float, sigma_px: float) -> np.ndarray:
    half = int(np.ceil(radius_px + 3 * sigma_px))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    return (-np.exp(-0.5 * ((np.hypot(yy, xx) - radius_px) / sigma_px) ** 2)).astype(np.float32)

# ring_ncc: best normalised cross-correlation over a radius sweep, with the radius (voxels) that gave it
def ring_ncc(img: np.ndarray, radii_px: np.ndarray, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    best = np.full(img.shape, -1.0, dtype=np.float32)
    radius = np.zeros(img.shape, dtype=np.float32)
    for r in radii_px:
        ncc = match_template(img, ring_template(float(r), sigma_px), pad_input=True)
        better = ncc > best
        best[better] = ncc[better]
        radius[better] = r
    return best, radius

# radius_sweep: radii in voxels covering the configured cage range
def radius_sweep(config: Config, pixel_size_a: float) -> np.ndarray:
    lo, hi = config.cage_radius_range_a
    return np.arange(lo, hi + 1e-6, config.radius_step_a) / pixel_size_a

# ring_snr: dark-ring contrast over background noise for a ring of radius_px centred at (y, x), valid voxels only
def ring_snr(img: np.ndarray, valid: np.ndarray, centre: tuple[float, float], radius_px: float, config: Config, pixel_size_a: float) -> float:
    d = _distance(img.shape, centre)
    ring = valid & (np.abs(d - radius_px) <= config.ring_halfwidth_a / pixel_size_a)
    bg = valid & (d >= config.bg_range[0] * radius_px) & (d <= config.bg_range[1] * radius_px)
    # too little of either annulus inside the volume to say anything
    if ring.sum() < 10 or bg.sum() < 10:
        return 0.0
    return float((img[bg].mean() - img[ring].mean()) / (img[bg].std() + 1e-6))

# walk_z: grow a symmetric slab about slice zc one slice at a time, returning the step with the best ring SNR
def walk_z(sub: np.ndarray, valid: np.ndarray, zc: int, config: Config, pixel_size_a: float) -> Step:
    px = pixel_size_a
    radii = radius_sweep(config, px)
    sigma = config.ring_sigma_a / px
    kmax = round(config.walk_max_half_a / px)
    centre = (sub.shape[1] // 2, sub.shape[2] // 2)
    centre = (float(centre[0]), float(centre[1]))
    slab = sub[zc].copy()
    sweep = radii
    best: Step | None = None
    misses = 0
    for k in range(kmax + 1):
        if k:
            lo, hi = zc - k, zc + k
            # the slab has run out of volume on both sides
            if lo < 0 and hi >= sub.shape[0]:
                break
            for zi in (lo, hi):
                if 0 <= zi < sub.shape[0]:
                    slab = np.minimum(slab, sub[zi])
        img = _prepare(slab, config, px)
        y, x, r = _locate(img, centre, sweep, sigma, config, px)
        r = _radial_radius(img, valid, (y, x), r, config, px)
        step = Step(k, y, x, r, ring_snr(img, valid, (y, x), r, config, px))
        # later steps only need the radii next to the current one
        centre, sweep = (y, x), radii[np.argsort(np.abs(radii - r))[:3]]
        if best is None or step.snr > best.snr:
            best, misses = step, 0
        else:
            misses += 1
            if misses >= config.walk_patience:
                break
    return best

# estimate_candidate: walk the slab, scan the z centre, and report geometry and box for one pick
def estimate_candidate(vol: np.ndarray, cand: Pick, config: Config) -> Estimate:
    px = cand.pixel_size_a
    half = int(np.ceil((config.bg_range[1] * config.cage_radius_range_a[1] + config.locate_range_a) / px))
    kmax = round(config.walk_max_half_a / px)
    shift = round(config.z_shift_a / px)
    z, y, x = round(cand.z), round(cand.y), round(cand.x)
    sub, valid, z0 = _crop(vol, z - kmax - shift, z + kmax + shift, y, x, half)
    zc = z - z0
    best = walk_z(sub, valid, zc, config, px)
    dz, snr = _scan_z(sub, valid, zc, best, shift, config, px)
    # crop centre is index half, so crop coordinates map back by (value - half)
    ex, ey, ez = x + (best.x - half), y + (best.y - half), z + dz
    radius_a = best.radius_px * px
    flags = []
    if min(ex, ey) - best.radius_px < 0 or ex + best.radius_px > vol.shape[2] - 1 or ey + best.radius_px > vol.shape[1] - 1:
        flags.append('edge')
    if zc + dz - best.k <= 0 or zc + dz + best.k >= sub.shape[0] - 1:
        flags.append('zedge')
    if best.k >= kmax:
        flags.append('zmax')
    return Estimate(
        tomogram=cand.tomogram,
        x=ex,
        y=ey,
        z=float(ez),
        pixel_size_a=px,
        radius_a=radius_a,
        box_a=2 * radius_a * config.box_margin,
        snr=snr,
        z_lo=ez - best.k,
        z_hi=ez + best.k,
        flag=','.join(flags),
    )

# write_estimates: dump estimates to CSV
def write_estimates(ests: list[Estimate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Estimate.model_fields))
        writer.writeheader()
        writer.writerows(e.model_dump() for e in ests)

# read_estimates: load estimates from CSV
def read_estimates(path: Path) -> list[Estimate]:
    with open(path, newline='') as handle:
        return [Estimate(**row) for row in csv.DictReader(handle)]

# run_estimate: estimates every pick in out/picks.csv and writes out/estimates.csv
def run_estimate(config_path: Path) -> None:
    config = load_config(config_path)
    picks = read_picks(config.out_dir / 'picks.csv')
    out: list[Estimate] = []
    for name, tomo in find_tomograms(config).items():
        mine = [p for p in picks if p.tomogram == name]
        if not mine:
            continue
        vol = load_volume(tomo, source=config.source)
        for pick in mine:
            est = estimate_candidate(vol, pick, config)
            out.append(est)
            print(
                f'{name}  x={est.x:7.1f} y={est.y:7.1f} z={est.z:6.1f}  r={est.radius_a:5.0f} A  box={est.box_a:5.0f} A'
                f'  snr={est.snr:5.2f}  slab {est.z_lo}-{est.z_hi}  {est.flag}'
            )
    write_estimates(out, config.out_dir / 'estimates.csv')

# _distance: distance of every pixel from a (y, x) centre
def _distance(shape: tuple[int, ...], centre: tuple[float, float]) -> np.ndarray:
    yy, xx = np.indices(shape)
    return np.hypot(yy - centre[0], xx - centre[1])

# _prepare: smoothed, background-flattened minimum projection
def _prepare(slab: np.ndarray, config: Config, pixel_size_a: float) -> np.ndarray:
    return highpass(smooth(slab, config.smooth_sigma_a / pixel_size_a), config.highpass_sigma_a / pixel_size_a)

# _locate: (y, x, radius_px) maximising ring NCC within locate_range of the previous centre
def _locate(img: np.ndarray, centre: tuple[float, float], radii: np.ndarray, sigma: float, config: Config, pixel_size_a: float) -> tuple[float, float, float]:
    ncc, radius = ring_ncc(img, radii, sigma)
    ncc = np.where(_distance(ncc.shape, centre) <= config.locate_range_a / pixel_size_a, ncc, -np.inf)
    y, x = np.unravel_index(np.argmax(ncc), ncc.shape)
    return float(y), float(x), float(radius[y, x])

# _radial_radius: radius (voxels) of the darkest annulus within one radius step of r0, from the 1-voxel radial profile
def _radial_radius(img: np.ndarray, valid: np.ndarray, centre: tuple[float, float], r0: float, config: Config, pixel_size_a: float) -> float:
    idx = np.rint(_distance(img.shape, centre)).astype(int)[valid]
    profile = np.bincount(idx, weights=img[valid]) / np.maximum(np.bincount(idx), 1)
    reach = config.radius_step_a / pixel_size_a
    lo, hi = max(int(r0 - reach), 1), min(int(r0 + reach) + 1, len(profile))
    return float(lo + np.argmin(profile[lo:hi]))

# _scan_z: slab centre shift (slices) maximising ring SNR for the best half-thickness and ring, with that SNR
def _scan_z(sub: np.ndarray, valid: np.ndarray, zc: int, best: Step, shift: int, config: Config, pixel_size_a: float) -> tuple[int, float]:
    scan = (0, best.snr)
    for dz in range(-shift, shift + 1):
        lo, hi = max(zc + dz - best.k, 0), min(zc + dz + best.k, sub.shape[0] - 1)
        img = _prepare(sub[lo : hi + 1].min(axis=0), config, pixel_size_a)
        snr = ring_snr(img, valid, (best.y, best.x), best.radius_px, config, pixel_size_a)
        if snr > scan[1]:
            scan = (dz, snr)
    return scan

# _crop: sub-volume (z range clipped) round (y, x) with edge padding, the xy mask of real voxels, and the first z index
def _crop(vol: np.ndarray, z0: int, z1: int, y: int, x: int, half: int) -> tuple[np.ndarray, np.ndarray, int]:
    z0, z1 = max(z0, 0), min(z1, vol.shape[0] - 1)
    ys = slice(max(y - half, 0), min(y + half + 1, vol.shape[1]))
    xs = slice(max(x - half, 0), min(x + half + 1, vol.shape[2]))
    block = np.asarray(vol[z0 : z1 + 1, ys, xs], dtype=np.float32)
    pad_y = (ys.start - (y - half), (y + half + 1) - ys.stop)
    pad_x = (xs.start - (x - half), (x + half + 1) - xs.stop)
    valid = np.pad(np.ones(block.shape[1:], dtype=bool), (pad_y, pad_x))
    return np.pad(block, ((0, 0), pad_y, pad_x), mode='edge'), valid, z0
