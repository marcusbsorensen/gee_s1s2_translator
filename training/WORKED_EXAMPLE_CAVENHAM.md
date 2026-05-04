# Worked example — Cavenham Heath SSSI, Suffolk Brecks

A demonstration that the trained operational model (`v2_equivalent_initial`)
produces sensible predicted reflectance over a UK lowland-heath site
that is **outside its training distribution**. Cavenham Heath sits in
the Suffolk Brecks (~52.29° N, 0.59° E), roughly 100 km north-east of
the nearest training AOI (Ashdown Forest, East Sussex) and on a
different Sentinel-2 tile (T31UCT vs T30U* for the entire training set).

![Cavenham Heath worked example: actual Sentinel-2 RGB next to U-Net predicted RGB, 26 June 2024](../docs/images/worked_example_cavenham.png)

## Why this site

- **SSSI / NNR lowland heath**, comparable habitat type to the training
  AOIs (heather, bracken, gorse, scattered Scots pine), so the
  distinction is regional rather than ecological — exactly the kind
  of generalisation an NGO operator at a site we didn't train on
  would need.
- **Brecks region** is climatically and pedologically distinct from
  the southern-England training set: thinner soils, more
  continental rainfall pattern, slightly later phenology.
- **Sentinel-2 tile T31UCT** is at the same UTM-zone boundary as the
  Brecks AOIs, so the Sentinel-1 incidence-angle distribution differs
  from training.
- 26 June 2024 — peak summer growth, cloud-free; no recent fire
  visible at the site, so this becomes a "predicted vs truth" check
  rather than a fire-mapping demo.

## Method

- AOI defined as a 2 000 m point-buffer around the heath centroid
  (52.291° N, 0.587° E) — exactly the operator-facing harvest path
  documented in `OPERATIONAL_DEPLOYMENT.md`'s Workflow 3.
- Phase 1 harvest run via the local CLI:
  `python -m gee_s1s2.cli harvest --aoi "Cavenham Heath" --window "comparison 2024"`.
  20 patches landed at `gs://.../patches/test/cavenham-heath/`,
  spanning four cloud-free S2 dates in 2024.
- Inference run via Workflow 1:
  `python scripts/submit_predict_aois.py --target cavenham-heath:20240626`
  with `save_truth: True` so the actual Sentinel-2 mosaic is written
  alongside the predicted one for direct comparison.
- Metrics computed over the 81 996 jointly-valid pixels (51 % of
  the AOI mosaic; the remainder is either Sentinel-2 cloud-mask zeros
  or patch-coverage gaps, masked out of the metric in both rasters).

## Per-band results vs the Ashdown out-of-distribution baseline

| Band | Cavenham MAE | Cavenham variance retention | Ashdown MAE (baseline OOD) | Ashdown variance retention |
| --- | ---: | ---: | ---: | ---: |
| B02 (blue) | 0.023 | 31 % | 0.015 | 48 % |
| B03 (green) | 0.035 | 34 % | 0.018 | 55 % |
| B04 (red) ⓓ | 0.053 | **21 %** | 0.021 | **43 %** |
| B08 (NIR) ⓓ | 0.093 | **71 %** | 0.093 | **60 %** |
| B11 (SWIR-1) ⓓ | 0.088 | **60 %** | 0.042 | **63 %** |
| B12 (SWIR-2) ⓓ | 0.097 | **28 %** | 0.030 | **56 %** |
| **Driver mean** | — | **45.1 %** | — | **55.5 %** |
| **Overall MAE** | **0.065** | — | — | — |

ⓓ = driver band; the operational pass bracket is 75–105 % retention.

## Qualitative read

Looking at the figure: the predicted RGB is recognisably the same
landscape as the truth panel — field boundaries, the heath / arable
mosaic, scattered woodland — but at lower contrast and slightly
flattened colour balance. The prediction smooths over the high-
frequency texture that the truth has, especially in the bright
field interiors.

The numbers say the same thing more precisely:

- **B08 (NIR) variance retention 71 %** is within striking distance of
  the 75 % operational bracket and is *better* on Cavenham than on
  Ashdown (60 %). NIR is the band that drives NDVI and NBR, so this
  is the operationally important one.
- **B04 (red) and B12 (SWIR-2) collapse hard** — 21 % and 28 %
  retention respectively, materially worse than Ashdown's 43 % and
  56 %. These bands carry the bare-soil and burn-discrimination
  signal, so dNBR thresholds fitted on training data will need
  re-calibrating for Cavenham.
- **MAE roughly doubles versus Ashdown**: 0.065 overall here vs
  ~0.030 on Ashdown. The model still produces reflectance in the
  right range and the right *spatial layout* — but the per-pixel
  fidelity halves when the regional regime differs.

## Operational verdict

The model is **operationally credible for perimeter delineation** at
Cavenham — the predicted scene reads as the right landscape. It is
**not yet usable for severity classification or absolute reflectance
analysis** at this site without local re-calibration of thresholds.
The recommended path for an NGO operator at a Brecks-region site:

1. Run inference at the site for a known cloud-free date (Workflow 1).
2. Compare the predicted RGB qualitatively against an actual
   Sentinel-2 tile from the same date — the prediction should look
   like the right place.
3. Empirically fit a dNBR threshold using a known-good fire perimeter
   from the same regional regime (the SWT-mapped Brentmoor / Poors
   Allotment perimeters in this repo are the closest available
   examples; ideally a Brecks-region fire perimeter would be used).
4. Apply the calibrated threshold to predicted dNBR for new sites.

If accuracy is critical (e.g. compensation cases), the right move is
**Workflow 4 — re-train including Brecks AOIs**. Adding 2–3 SSSI
heath sites from the region (Roydon Common, Cavenham Heath itself,
Knettishall Heath) plus a held-out Brecks test site would let the
trained model see the regional regime explicitly. Cost ~£0.50 for
the harvest plus another £0.50 for training; one-day work.
