# gee_s1s2_translator — Phase 1 (harvesting)

Sentinel-1 to Sentinel-2 translation for lowland heath fire mapping,
running on Google Earth Engine and Google Cloud Storage.

This is a port of the v2 PyTorch + Microsoft Planetary Computer pipeline
(see `../s1s2-translator/Sonia_Project_Briefing.md` for the dissertation
context) to a fully-Google-managed stack so you can run it operationally
under your own Google account once it's handed over. Phase 1 is the
harvesting layer; training (Colab) and inference deployment (Vertex AI)
are Phases 2 and 3, with operational packaging in Phase 4.

The user of this README is **you, Sonia**. It assumes you have a
working laptop, a Google account, and roughly an hour to set up a fresh
Earth Engine project and a Cloud Storage bucket. None of the steps need
ML or remote-sensing experience; the technical depth is in the code.

## What this repository does

When run end to end, this repository:

1. Builds 16 areas of interest (AOIs): two fire perimeters from your
   Surrey Wildlife Trust KML, one wider TBH SPA polygon, two fire-area
   training buffers, ten new heath training sites across Surrey / the
   New Forest / the Dorset Heaths, the held-out Ashdown Forest test
   site, and the Hankley Common sanity-check site.
2. Searches Earth Engine for matched pairs of Sentinel-1 SAR and
   Sentinel-2 optical acquisitions over each AOI in four cloud-free
   date windows, plus the post-fire 2022 inference window.
3. Calibrates each Sentinel-1 scene to gamma-naught dB using the
   community-standard recipe (Mullissa et al. 2021), so the values are
   numerically comparable to the v2 results.
4. Filters out cloudy scenes by computing each scene's cloud cover
   inside the AOI mask (not the whole tile).
5. Samples 256x256 paired patches per scene with a deterministic seed
   for reproducibility.
6. Exports the patches to your Cloud Storage bucket as TFRecord, with a
   manifest CSV recording every paired scene used.

The output is a Cloud Storage layout that the Phase 2 training notebook
will read from directly in Colab.

## Setup

This is a one-time process. After it's done, you don't need to think
about it again.

### Step 1: Earth Engine

1. Go to https://code.earthengine.google.com and sign in with your
   Google account. If you have not registered before, you'll be
   prompted; non-commercial use is free.
2. Create or pick a Cloud project. Note its project ID (looks like
   `my-project-name-12345`). Earth Engine asks for this on first use.
3. Run `earthengine authenticate` from your terminal once. This opens
   a browser window, you sign in, and a small credentials file gets
   stored in `~/.config/earthengine/`.

### Step 2: Cloud Storage bucket

1. Open the [Cloud Console](https://console.cloud.google.com/) and pick
   the same project you used for Earth Engine.
2. Open Cloud Storage → Buckets → Create bucket.
3. Pick a name. Bucket names are globally unique, so something like
   `<your-name>-heath-fire-mapping` works.
4. Storage class: **Standard**. Location: **europe-west2** (London) is
   sensible for UK data.
5. Leave the rest as defaults.
6. Run `gcloud auth application-default login` from your terminal once
   so the Python client can read from your bucket without a service
   account key. (Install `gcloud` from
   https://cloud.google.com/sdk/docs/install if you don't have it.)

### Step 3: Local install

```bash
git clone <this repo>
cd gee_s1s2_translator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[notebook]"
cp .env.example .env
```

Edit `.env`:

```
GEE_PROJECT_ID=<your project id from Step 1>
GCS_BUCKET=<your bucket name from Step 2>
GCS_PREFIX=gee_s1s2_translator
```

Leave `GOOGLE_APPLICATION_CREDENTIALS` blank for now; the application
default credentials from `gcloud auth application-default login` are
fine for getting started.

### Step 4: Verify

```bash
gee_s1s2 init --config config/operational_v1.yaml
```

You should see green ticks for Config / GEE / GCS. If any of them
fails, the error message points at the specific thing to fix.

## Running the pipeline

After init passes:

```bash
# Highest-risk step: validate that the GEE S1 calibration matches MPC.
# Reads three (AOI, date) reference samples from the v2 archive at
# ../s1s2-translator/data/runs/v2_diverse_heath/manifest.csv.
gee_s1s2 calibrate_check --config config/operational_v1.yaml

# Dry run: candidate counts per (AOI, window). No exports.
gee_s1s2 harvest --config config/operational_v1.yaml --dry-run

# Real harvest: exports TFRecords to gs://<bucket>/<prefix>/operational_v1/
# This takes 30-90 minutes depending on GEE export queue depth.
gee_s1s2 harvest --config config/operational_v1.yaml

# Manifest summary, in the same format as the v2 PyTorch project.
gee_s1s2 manifest summary --config config/operational_v1.yaml
```

## What the output looks like

After a full harvest, your bucket has:

```
gs://<your-bucket>/gee_s1s2_translator/operational_v1/
├── manifest.csv                                # one row per (AOI, S1 scene, S2 scene)
└── patches/
    ├── train/<aoi-slug>/<aoi-slug>__<window-slug>-00000.tfrecord.gz
    ├── val/<aoi-slug>/...
    └── test/<aoi-slug>/...
```

Each TFRecord example is one 256x256 patch with eight float32 bands:

* S1: VV, VH (calibrated gamma-naught in dB)
* S2: B02, B03, B04, B08, B11, B12 (reflectance, scaled to [0, 1])

This format is what the Phase 2 Colab training notebook reads directly.

## Verifying the TFRecord output

The notebook `notebooks/01_harvest_walkthrough.ipynb` walks through this
step by step. The short version:

```python
from gee_s1s2.export import open_tfrecord_local
ds = open_tfrecord_local(
    'local_cache/sample.tfrecord.gz',
    bands=['VV', 'VH', 'B02', 'B03', 'B04', 'B08', 'B11', 'B12'],
)
for example in ds.take(1):
    print({k: v.shape for k, v in example.items()})
```

You should see `{ 'VV': (256, 256), 'VH': (256, 256), 'B02': (256, 256), ... }`.

## Costs

Negligible during the build phase, low during operational use. See
`docs/cost_monitoring.md` for the full budget projections including
Vertex AI inference deployment options (Phase 3).

## Cost-of-mistakes notes

* GEE compute and exports are free under non-commercial registration but
  rate-limited. If your harvest queue takes hours, that's normal.
* Cloud Storage is roughly £0.02/GB/month at standard tier. The full
  v2-equivalent harvest is around 1.5 GB. Cost is therefore pence per
  month even at scale.
* Service account JSON keys, if you choose to use them, must never be
  committed to git. The `.gitignore` covers `*-key.json` and
  `service-account*.json` patterns.

## Where to look next

* `docs/architecture_overview.md` — how this maps to the v2 PyTorch
  pipeline and to Phases 2-4.
* `docs/calibration_methodology.md` — the S1 calibration recipe with
  references and the GEE-vs-MPC validation results.
* `docs/cost_monitoring.md` — operational cost projections including
  the three Vertex AI inference architectures.
* `notebooks/01_harvest_walkthrough.ipynb` — interactive walkthrough.

## Phase boundaries

This repository is **Phase 1 only** (harvesting to TFRecord on GCS).

* Phase 2: training in Google Colab, reading TFRecords from this bucket.
* Phase 3: inference deployment on Vertex AI.
* Phase 4: operational packaging and handover documentation.

Each phase has its own kickoff and review gate. Don't run any further
work until Phase 1's review gate has cleared.

## Attribution

This is the GEE port of the v2 lowland heath fire mapping pipeline,
which was built collaboratively with Marcus Sorensen using Claude Code.
For the broader project history (v1, v1.5, v2 results, methodology
decisions), see `../s1s2-translator/Sonia_Project_Briefing.md`.
