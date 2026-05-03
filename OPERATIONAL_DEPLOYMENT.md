# Operational deployment — `gee_s1s2_translator`

This document is written for **wildlife protection officers, conservation
NGO staff, and similar operationally-motivated readers** who want to know:
what this tool does, whether it's likely to work for their site, what it
will cost, and how to deploy it. It is deliberately not academic — the
methodology paper is in `docs/`, the dissertation grew this work, and the
reference numbers live in `validation_report.md`.

If you came here looking for "I have a heath that burned, can I see the
damage map even though it's been raining for three weeks?", read on.

---

## What this tool does

`gee_s1s2_translator` is a deep-learning model that **predicts what a
Sentinel-2 optical image of a UK lowland heath would look like, from a
Sentinel-1 radar image of the same scene**. Sentinel-1 sees through
clouds. Sentinel-2 doesn't. So the model lets you reason about heath
vegetation state — including post-fire scarring — on dates where the only
real Sentinel-2 image available is a flat grey cloud sheet.

The output is a 9-band GeoTIFF: six predicted Sentinel-2 reflectance
bands (B02, B03, B04, B08, B11, B12), plus the three derived indices
NDVI, NBR, NDWI computed from them. NBR is typically the most informative
band for heath fire damage; threshold it the same way you'd threshold a
real Sentinel-2 NBR raster.

## What use cases it solves

The motivating use case: **mapping fire scar boundaries on lowland heath
when cloud cover prevents direct Sentinel-2 observation.** Lowland heath
fire seasons are getting longer in the UK, and Atlantic-margin
weather means useful post-fire optical imagery is the exception rather
than the rule. Field-mapping a perimeter with GPS is excellent ground
truth but slow and resource-intensive; this tool augments it by giving
operators a quick desk-side approximation of what NBR-thresholded fire
scar maps would show on cloud-covered dates.

The same pattern transfers to:

- **Vegetation recovery monitoring** through cloudy autumns and winters
- **Pre-fire condition snapshots** when a fire is already underway and
  the most recent cloud-free Sentinel-2 is weeks old
- **Comparing damage extent** between a known-healthy spring scene and a
  predicted post-fire scene, even when the post-fire date has no usable S2

It is a translation tool, not a classifier. It does not output
"burned / not burned" labels — you make that call by thresholding the
predicted reflectance bands or the predicted NBR, the same way you would
on a real Sentinel-2 image. The validation report quantifies how close
the predicted bands come to truth on a held-out test set; the answer is
"close enough to be useful for thresholding, with the caveats below."

## What it requires to deploy

A working operational deployment needs five things, all free or
trivially-priced for the scale this tool runs at:

1. **Google Cloud account.** A free-tier account with billing enabled
   (no credit needed up front). All compute and storage runs in your
   own GCP project.
2. **Earth Engine non-commercial registration.** Free. Sign up at
   <https://signup.earthengine.google.com>. Approval is fast for
   non-commercial users. Once approved, the project ID you register is
   what the harvest pipeline uses.
3. **Google Cloud Storage bucket.** One bucket in a single region (we
   use `europe-west2` because that's where our compute runs; pick a
   region close to your AOIs to minimise data transfer). Free-tier
   storage covers comfortably more than this tool produces.
4. **Vertex AI T4 GPU quota of 1.** Vertex's default for new projects
   is 0; you need to request `Custom model training Nvidia T4 GPUs
   per region` raised to 1 in your chosen region. Approval is usually
   automatic and within minutes for value-of-1 requests. The project's
   training and inference runs all use a single T4 at a time.
5. **A list of AOIs (areas of interest) you care about.** Either a KML
   file with each fire perimeter or heath polygon, or simple lat/lng
   centre points with a buffer radius. Any GIS tool can produce these.

That's it. No on-prem GPUs, no commercial licences, no data sharing
agreements. The model weights and code are MIT-licensed (see below).

## What it costs to operate

Concrete numbers from the project's actual deployment, May 2026:

| Step | Vertex AI compute | Wall-clock | Cost (£) |
| --- | --- | ---: | ---: |
| Train model from scratch on a fresh AOI set | Custom Job, 1× T4 on n1-standard-4 | ~20 min | £0.30–0.60 |
| Phase 1 harvest (one-off + per new AOI) | Custom Job, n1-standard-4 CPU | ~1.5–2.5 h | £0.30–0.50 |
| Single inference run over 3 AOIs | Custom Job, 1× T4 | ~3 min | £0.02 |
| Vertex AI Endpoint, *while running* | n1-standard-2 + 1× T4 | per hour | £0.20–0.50 |
| GCS storage for harvested patches | per GB-month | per month | <£0.05 |

**Headline:** training a model and running inference is well under £1.
The cost lever is the Vertex AI Endpoint — it bills by the hour while
deployed, even when no one's calling it.

**Operational recommendation: deploy-and-tear-down rather than always-on.**
A typical NBR-threshold workflow needs the endpoint for ~10 minutes total
(deploy → predict → tear down), so the marginal cost is pennies. Leaving
an endpoint running 24/7 for a year is ~£1,750–4,400 — only worth it if
you're running predictions multiple times per hour across many sites.

The repo's training README documents the offline-inference path
(`predict_aois.py` as a Vertex Custom Job) which avoids endpoint costs
entirely and is the right choice for batch evaluation.

## How to deploy

Three high-level workflows, depending on what you want to do.

### Workflow 1: predict over our existing AOIs

If your AOIs overlap with the project's existing harvest (Brentmoor
Heath, Poors Allotment, the Surrey/New Forest/Dorset training sites, or
Ashdown Forest), the trained model is already published and you can
predict directly.

```bash
# Submit an inference Vertex Custom Job (writes 9-band GeoTIFFs to GCS).
python scripts/submit_predict_aois.py
```

Outputs go to `gs://<bucket>/<prefix>/operational_v1/models/v2_equivalent_initial/predictions/unet/`.

### Workflow 2: predict over a new AOI

Edit `config/operational_v1.yaml`, add your AOI under `aois:`, set its
`role: target` if it's a fire perimeter you want to predict over (or
`role: training` if you're harvesting it for a future re-train).
Re-run the Phase 1 harvest restricted to your new AOI:

```bash
# Local CLI
python -m gee_s1s2.cli harvest --only-aoi "Your Site Name"

# Or as a Vertex Custom Job (recommended for unattended runs > 30 min)
python scripts/submit_post_fire_harvest.py
```

Then re-run inference as in Workflow 1.

### Workflow 3: re-train on a different region

Rare — only needed if your sites are far enough away from southern
England that vegetation phenology and S1 calibration regimes differ
materially. The harvest config supports any AOI definition. You'd:

1. Add new AOIs (yours plus a held-out test AOI in the same regional
   biome) under `aois:`.
2. Run Phase 1 harvest to populate `gs://<bucket>/<prefix>/patches/`.
3. Run Phase 2 training: `python scripts/submit_vertex_training.py` —
   this fits a fresh model from your harvest. Expected ~£0.30–0.60.
4. Validate against your held-out AOI; compare to v2/this run's metrics
   in `validation_report.md`.

The end-to-end is single-day work for an analyst familiar with GCP.

## How to interpret the outputs

Each predicted GeoTIFF is 9 bands, in this order:

| Band | Name | Meaning | Operational use |
| ---: | --- | --- | --- |
| 1 | B02 | Sentinel-2 blue (492 nm) | Atmospheric reasoning, water |
| 2 | B03 | Sentinel-2 green (560 nm) | Live vegetation, water |
| 3 | B04 | Sentinel-2 red (665 nm) | Bare soil, ash, fire scarring |
| 4 | B08 | Sentinel-2 NIR (842 nm) | Live vegetation chlorophyll |
| 5 | B11 | Sentinel-2 SWIR-1 (1610 nm) | Burnt-area discrimination |
| 6 | B12 | Sentinel-2 SWIR-2 (2190 nm) | Burnt-area discrimination |
| 7 | NDVI | (B08–B04) / (B08+B04) | Live vegetation index |
| 8 | **NBR** | (B08–B12) / (B08+B12) | **Normalised Burn Ratio** |
| 9 | NDWI | (B03–B08) / (B03+B08) | Water/moisture index |

**NBR is the workhorse band for fire-scar mapping**; standard practice
is to threshold pre-fire NBR minus post-fire NBR (`dNBR`) to delineate
the perimeter. The same workflow runs against this tool's outputs as
against real Sentinel-2 — the GeoTIFFs are 10 m UTM rasters,
DEFLATE-compressed, with proper CRS and affine.

**Honest caveat from the validation report:** on the held-out Ashdown
test set, the U-Net retains **55.5 %** of patch-specific reflectance
variance on driver bands (B04 / B08 / B11 / B12) — below the [75 %, 105 %]
operational pass bracket. The pattern is consistent with regression to
the mean: the model finds a smoother solution that minimises absolute
error without preserving high-frequency variance. **This means
NBR-threshold values that are calibrated against real Sentinel-2 may
need re-tuning when applied to predicted Sentinel-2.** The variance
collapse is uniform across our test set, so the right operational
practice is: run dNBR thresholding on a known-good fire perimeter (e.g.
Brentmoor 2022 from this project's predictions) against ground truth,
empirically derive a threshold offset, then apply that offset to new
sites with the same calibration regime.

## Provenance and methodology

The deep-learning component is a 4-level U-Net (~7.8 M parameters) with
bilinear upsampling and a sigmoid output, trained with combined L1 +
0.5 × L2 loss using Adam at 1e-4 and early stopping on validation RMSE.
Architecture is byte-for-byte the v2 PyTorch reference — see
`docs/methodology_divergences.md` for the three documented divergences
between this GEE-port pipeline and the v2 PyTorch + Microsoft Planetary
Computer pipeline:

1. Sentinel-1 calibration uses the Vollrath/Mullissa volumetric
   gamma-naught model in Earth Engine, ~1.5 dB below MPC's ESA range-
   Doppler RTC. Internally consistent and not a bug; documented in
   `docs/calibration_methodology.md`.
2. Held-out Ashdown test count is 105 patches in this run vs the v2
   reference's 76, due to GEE's random-origin sampler with 50 %
   overlap fitting more origins per scene than v2's deterministic grid.
3. T30UWB tile partial-coverage handling at Beaulieu Heath.

Headline numbers from `validation_report.md`:

- Best validation RMSE: **0.0769** at epoch 55 (vs v2 reference 0.1062)
- Driver-band variance retention: 55.5 % (vs v2 58.3 %; both fail
  the 75–105 % operational bracket consistently with each other)
- Per-band Ashdown MAE: B02 0.0151, B03 0.0181, B04 0.0211, B08 0.0929,
  B11 0.0418, B12 0.0302

## Citations and acknowledgements

The Sentinel-1 calibration recipe follows **Mullissa et al. (2021)**,
*"Sentinel-1 SAR Backscatter Analysis Ready Data Preparation in Google
Earth Engine"*, Remote Sensing 13(10), 1954. The Earth Engine
implementation draws on the open `adugnag/gee_s1_ard` repository.

The Sentinel-1 RTC convention difference (Vollrath/Mullissa volumetric
gamma-naught vs ESA range-Doppler) is discussed in **Vollrath, Mleczko,
Tracy (2020)**, *"Angular-Based Radiometric Slope Correction for
Sentinel-1 on Google Earth Engine"*, Remote Sensing 12(11), 1867.

The original v2 PyTorch + MPC reference implementation (`v2_diverse_heath`)
is the methodological starting point for the architectural and loss
choices reproduced here.

The dissertation work that this operational pipeline was built around —
*Lasocki (2026), University of [...], MSc dissertation* — should be
cited if you use this tool for academic publication.

This work was supported by **Surrey Wildlife Trust** field-mapped fire
perimeter data. The two 2022 fire perimeters used as primary inference
targets (Brentmoor Heath, Poors Allotment) are derived from SWT's
mapped fire database.

## Licence

MIT. See `LICENSE`. Forks and contributions are welcome — the
config-driven design is specifically intended to make adapting to new
regions and AOI sets a YAML edit rather than a code fork.

## Limitations

**Honest list. Read this before deploying operationally.**

- **Dataset scale.** The training set is **377 patches** (≈ 24 GB of
  Sentinel-2 imagery, sampled with overlap from 13 training-role AOIs
  in southern England 2021–2024). This is small by deep-learning
  standards. The model generalises within the southern-UK lowland
  heath biome it was trained on; do not expect it to work on Scottish
  blanket bog or Mediterranean garrigue without re-training.
- **Variance retention below the operational floor on driver bands.**
  See the interpretation paragraph above. The practical implication is
  that NBR thresholds need empirical re-calibration per site rather
  than copying values from real-Sentinel-2 thresholding workflows.
- **Single-region calibration.** Sentinel-1 backscatter regimes vary
  with vegetation type, soil moisture climate, and incidence angle.
  This model was calibrated on UK lowland heath at 50–51° N. The
  Vollrath terrain-flattening helps but does not eliminate regime
  differences.
- **Temporal scope.** Training data spans 2021–2024. Cycles longer
  than 4 years (e.g. multi-year drought recovery, decadal succession)
  are not represented in the training set.
- **Cloud-cover override is liberal in the post-fire 2022 window.**
  We accept any S2 scene as the pairing target for that window
  regardless of cloud cover (the operational point — see
  `config/operational_v1.yaml`'s `cloud_cover_override_pct: 100`).
  Predictions from these patches use only the S1 channels and are
  evaluable visually against ground-truth perimeters; quantitative
  per-pixel error against truth is not measurable for cloud-covered
  scenes (truth is hidden behind the cloud).

## Service-account hygiene note

The project uses a dedicated `gee-harvester@<project>.iam.gserviceaccount.com`
service account for the Phase 1 harvest, with `roles/earthengine.writer`
+ `roles/storage.objectAdmin` on the project. The `objectAdmin` role
is project-wide rather than scoped to the bucket prefix; this should be
narrowed to a single-bucket conditional binding before this is treated
as a permanent operational pattern. The current grant is fine for a
single-operator, single-bucket project; it is too broad for a
multi-tenant or shared-bucket scenario.

## Where to go next

- `validation_report.md` — full metrics, per-band breakdowns, v2
  reference comparison
- `docs/methodology_divergences.md` — the three documented divergences
  from the v2 reference and why each one is operationally inert
- `docs/calibration_methodology.md` — S1 calibration recipe rationale
  and validation-against-MPC results
- `training/README.md` — Phase 2 training loop documentation
- `training/v2_reference_results.json` — the v2 PyTorch reference
  numbers we compare against
- `config/operational_v1.yaml` — the YAML that controls the whole
  pipeline; this is where you start when adapting to a new region
