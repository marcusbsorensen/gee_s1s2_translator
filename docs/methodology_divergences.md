# Methodology divergences: GEE port vs v2 PyTorch + MPC

This document records the systematic differences between the GEE
operational pipeline (this repo) and the v2 PyTorch reference
pipeline (`../s1s2-translator`). It is the place to look when
writing operational methodology for Sonia's professional practice or
the dissertation, so the comparison numbers can be defended without
ambiguity.

## 1. S1 calibration: ~1.5 dB systematic offset

**What differs.** GEE's `COPERNICUS/S1_GRD` ships sigma-naught only;
this repo applies the **Vollrath/Mullissa volumetric-scattering model**
(`gee_s1s2/calibration.py`) to convert to gamma-naught. MPC's
`sentinel-1-rtc` collection, used by v2, ships a product produced by
**ESA's range-Doppler terrain correction** pipeline (descended from
SNAP / GAMMA), which assumes surface/specular scattering with an
additional `sin(θ_local) / sin(θ_ref)` term.

**Magnitude.** Per the validation in
`docs/calibration_methodology.md`, the GEE-calibrated values run
**~1.4–1.8 dB below MPC RTC** uniformly across all three samples and
both polarisations. Std ratios sit in [0.71, 0.91] (within the
[0.7, 1.3] pass criterion). DEM choice (SRTM 30 m vs Copernicus DEM
GLO-30) shifts the offset by < 0.05 dB on lowland heath, ruling DEM
out as a contributor.

**Operational implication.** The offset is **uniform**, so training
data is internally consistent and the U-Net learns its own input
distribution. If a paper or report needs MPC-aligned dB values,
post-process with `+1.5 dB` on both polarisations as a constant
correction.

## 2. Ashdown held-out test: 105 patches (GEE) vs 76 (v2)

**What differs.** Ashdown Forest is the held-out test AOI for the
generalisation argument (force-split to test in both pipelines). The
v2 archive shows 76 Ashdown patches; the GEE port produces **105**
across the same four cloud-free windows.

**Why.** Two compounding causes, neither of them a bug:

1. **Random-sampling origin density.** The GEE port samples random
   patch origins per scene with the v2 min-spacing rule and
   `patch_overlap_pixels = 128` (50 % overlap allowed). v2's manifest
   was generated from a deterministic grid sampler that effectively
   admits fewer non-overlapping origins for the same AOI. In a
   moderately-sized AOI like Ashdown's 5 km × 5 km buffer, the
   stochastic sampler with 50 % overlap can fit several extra origins
   per scene before the AOI capacity binds.
2. **GEE vs MPC S2 scene availability.** The pre-fire 2022 window over
   Ashdown produced 10 candidate pairs in the GEE harvest, slightly
   more than v2 saw on MPC. Sentinel-2 SR_HARMONIZED is the same
   Copernicus collection but indexed differently in the two catalogues,
   and a small handful of scenes appear in one but not the other on
   any given window.

**Operational implication.** **Do not subsample to match v2's count.**
The U-Net receives 38 % more held-out test data on the GEE side, which
strengthens the generalisation argument rather than weakens it. Phase
2's validation report should compute Ashdown MAE/RMSE on the full
105-patch test set and footnote the divergence when comparing back to
v2's 76-patch numbers.

## 3. T30UWB partial-coverage handling at Beaulieu Heath

**What differs.** The harvest's per-AOI cloud check
(`gee_s1s2/filtering.aoi_cloud_pct`) sometimes returns `None` from
`reduceRegion` instead of a numeric percentage. This is a defensive
edge case: for Beaulieu Heath (centred at lon −1.450 — directly on
the T30UWB / T30UXB MGRS tile boundary), 10 S2 scenes from the
T30UWB tile had **partial coverage of Beaulieu's bbox** with too few
SCL pixels for the cloud check to compute a meaningful percentage. We
treat `None` as 100 % cloud and reject the scene. The full harvest
log shows 10 such warnings; **all 10 are Beaulieu**, none for Matley
or Ibsley (which sit fully inside T30UWB).

**Magnitude.** Diagnostic run (2026-05-02): for each of the 10 scenes,
the geometric intersection with Beaulieu's bbox covers between 3.5 %
and 25.1 % of the AOI. Of those, 7 scenes have **zero SCL pixels** in
the intersection (the tile boundary wraps the bbox but the scene's
actual image data doesn't fill it); the remaining 3 have 269-409 SCL
pixels — far below the ~16,400 SCL pixels (at SCL's native 20 m grid)
needed for a single 256 × 256 patch. **Zero recoverable patches.**

**Operational implication.** None. The defensive None-handling is
correct, no patches are silently dropped, and Beaulieu's 80 patches
in the manifest are unaffected. The New Forest region's 20.5 % share
in the GEE harvest (vs v2's 23.9 %) is meteorology + AOI-size
variance, not a tile-boundary artefact. If Beaulieu's polygon ever
moves further onto the T30UXB side of the boundary in a future config,
this issue resolves itself — the AOI would be fully inside one tile.

---

## Where each divergence is logged

| Divergence | Numeric details | Lives in |
|---|---|---|
| S1 calibration offset | mean Δ −1.42 to −1.79 dB, std ratio 0.71–0.91 | `docs/calibration_methodology.md` (validation block + offset section) |
| Ashdown patch surplus | 105 vs v2's 76 (+38 %) | `training/v2_reference_results.json` (footnote on the comparison table) |
| T30UWB partial-coverage | 10 None warnings, 0 recoverable patches | this file (above) and the harvest log's `WARNING gee_s1s2.harvest: Cloud check failed for ... T30UWB` lines |
