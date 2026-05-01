# Architecture overview: GEE port and the four-phase plan

## What changed from v2 (PyTorch + MPC) to operational (GEE + GCS)

| Concern | v2 PyTorch (MPC) | GEE port (operational v1) |
| --- | --- | --- |
| S1 source | MPC `sentinel-1-rtc` (already gamma-naught) | GEE `COPERNICUS/S1_GRD` (sigma-naught) + custom calibration |
| S2 source | MPC `sentinel-2-l2a` | GEE `COPERNICUS/S2_SR_HARMONIZED` |
| Harvest engine | `odc-stac` via `pystac-client` | Earth Engine ImageCollection → batch Export |
| Patch storage | Local GeoTIFFs in `data/patches/` | TFRecord shards in GCS |
| Manifest | `data/manifest.csv` (local) | `gs://<bucket>/<prefix>/operational_v1/manifest.csv` |
| Training | Local Python + PyTorch | Google Colab (Phase 2) reading TFRecords |
| Inference deployment | Local CLI (`s1s2 predict`) | Vertex AI endpoint (Phase 3) |
| Operational user | Marcus (build) → Sonia (handover) | Sonia (operational), with handover docs in Phase 4 |

Configuration is byte-for-byte the same on the AOI / date window /
sample budget axis, with three new blocks (calibration, export,
training_split). This is deliberate: anything Sonia learned from the
v2 config carries directly over.

## Logical flow

```
+----------------+     +-------------------+     +-------------------+
|  S1 GRD coll.  | --> |  calibration.py   | --> |                   |
|  (GEE)         |     |  Mullissa recipe  |     |   pairing.py      |
+----------------+     +-------------------+     |   filtering.py    |
                                                  |   sampling.py     |
+----------------+     +-------------------+     |                   |
|  S2 SR coll.   | --> |  filtering.py     | --> |                   |
|  (GEE)         |     |  AOI cloud, SCL   |     +---------+---------+
+----------------+     +-------------------+               |
                                                            |
                                                            v
                                                  +-------------------+
                                                  |   export.py       |
                                                  |   TFRecord to GCS |
                                                  +---------+---------+
                                                            |
                                                            v
                                          +------------------------------+
                                          | gs://<bucket>/.../patches/   |
                                          | gs://<bucket>/.../manifest   |
                                          +-------------+----------------+
                                                        |
                                       Phase 2 (Colab)  |  reads from here
                                       Phase 3 (Vertex) |
                                                        v
```

## Phase boundaries

* **Phase 1 (this repo)**: harvesting layer. Sentinel-1 calibration,
  Sentinel-2 filtering, paired patch sampling, TFRecord export, manifest.
* **Phase 2 (Colab notebook, separate repo)**: training. Reads TFRecord
  shards from this repo's GCS output. Same U-Net architecture as the
  v2 PyTorch project, ported to TensorFlow + Keras for Colab/Vertex AI
  ergonomics. Outputs a SavedModel back to the same bucket.
* **Phase 3 (Vertex AI)**: inference deployment. Wraps the SavedModel
  in a Vertex AI Endpoint. Runs inference on cloud-covered S1
  acquisitions over the fire perimeters and writes predicted GeoTIFFs
  to the bucket.
* **Phase 4 (operational packaging)**: handover documentation, teardown
  scripts, and cost monitoring guidance. Sonia takes the toolchain over
  under her own Google account.

Each phase has its own kickoff and review gate. Don't run any further
work until Phase 1's review gate has cleared.

## v2-to-Phase-1 module mapping

| v2 module (`s1s2-translator/src/s1s2/`) | Phase 1 equivalent |
| --- | --- |
| `config.py` | `gee_s1s2/config.py` (mirrors schema, adds calibration / export) |
| `utils.py` (KML, point_buffer) | `gee_s1s2/aois.py` |
| `catalog.py` (MPC + CDSE) | `gee_s1s2/catalogue.py` (S1 GRD + S2 SR) |
| (was MPC RTC, no calibration step) | `gee_s1s2/calibration.py` (Mullissa recipe) |
| `pairs.py` | `gee_s1s2/pairing.py` |
| `extract.py` (SCL, validity) | `gee_s1s2/filtering.py` |
| `patches.py` (random_origins) | `gee_s1s2/sampling.py` |
| (was local GeoTIFF write) | `gee_s1s2/export.py` (TFRecord to GCS) |
| `manifest.py` | `gee_s1s2/manifest.py` (GCS-backed CSV with same schema) |
| `cli.py` | `gee_s1s2/cli.py` |
| `harvest.py` | `gee_s1s2/harvest.py` |

## Things to know about Earth Engine that aren't obvious from v2

* All `ee.*` operations are server-side and lazy. Calling `getInfo()`
  materialises a result; everything else is a deferred computation.
  The Phase 1 code calls `getInfo()` only for collection sizes,
  pair-finding metadata, and the AOI cloud check.
* `ee.batch.Export` is asynchronous. The Phase 1 harvest submits one
  export task per (pair) and writes the manifest immediately so reruns
  can dedup. Tasks complete on Google's side over the next 30-90
  minutes; the Phase 2 training notebook polls bucket contents rather
  than the task list.
* GEE's `COPERNICUS/S1_GRD` does not include the LUT bands MPC RTC
  exposes. We compute the calibration coefficients from incidence
  angle and DEM slope server-side rather than pulling LUTs.

## Risk register for the GEE port

| Risk | Mitigation |
| --- | --- |
| Calibration drift vs MPC RTC | `calibrate_check` CLI command + `docs/calibration_methodology.md` validation block. Pass criterion: ±2 dB mean, std ratio within 30%. |
| GEE export queue throttling | Per-account quotas; document expected wall-clock and add `max_concurrent_tasks` config. |
| GCS access mis-configured | `init` command exits with a clear error; README's "Setup: GCS bucket" walks through `gcloud auth application-default login`. |
| `gee_s1_ard` upstream fixes drift from our reimplementation | Annual sync check; `docs/calibration_methodology.md` records the source-commit reference. |
| Schema drift between Phase 1 manifest and Phase 2 training loader | Phase 1 manifest schema is byte-for-byte v2's (plus three GEE-extras at the end). Phase 2 training loader reads via `pandas.read_csv` and is robust to extra columns. |
| Patch sampling: `gee_s1s2/sampling.py` exposes random-origin sampling (per v2's `random_patches` strategy), but `export.py` currently submits each pair to GEE with `patchDimensions=[256,256]`, which produces a *grid-tiled* output rather than random origins. For small AOIs (e.g. Brentmoor) this yields 1 patch per scene; for large AOIs the grid covers the bbox. Phase 2 design item: decide whether to keep grid-tiling (simpler, deterministic, fewer training samples) or wire sampling.py back in via per-origin `Export.image.toCloudStorage` clipped to a 256m × 256m sub-region (closer to v2 design, more samples, but `n_pairs × n_patches` export tasks and queue pressure). |
