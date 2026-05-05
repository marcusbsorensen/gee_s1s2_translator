"""Submit a Vertex inference job over a list of (AOI, S1 acquisition date) targets.

This is the operator-facing path for running the trained U-Net over
patches that are already in GCS (i.e. you have already harvested them
via Phase 1). For each target, it loads the model, runs a forward pass
on every matching patch, and writes a 9-band GeoTIFF (B02-B12 + NDVI/
NBR/NDWI) to GCS.

Cost: ~£0.02 for a typical run of 1-3 AOIs. T4 + n1-standard-4 for
~5 minutes. The T4 is required only because Vertex's tf-gpu prebuilt
image refuses to start without an accelerator; the actual prediction
work is small.

Examples
--------

# Predict over the project's existing test sites (defaults)
python scripts/submit_predict_aois.py

# A single target
python scripts/submit_predict_aois.py \\
    --target brentmoor-area-training:20221008:postfire

# Multiple targets, custom run name + output sub-directory
python scripts/submit_predict_aois.py \\
    --target brentmoor-area-training:20221008:postfire \\
    --target poors-allotment-area-training:20221008:postfire \\
    --output-subdir post_fire_2022 \\
    --run-name v2_equivalent_initial
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_TARGETS = [
    {"aoi_slug": "brentmoor-area-training", "s1_date": "20230526",
     "out_label": "brentmoor-area-training_20230526", "save_truth": False},
    {"aoi_slug": "poors-allotment-area-training", "s1_date": "20230526",
     "out_label": "poors-allotment-area-training_20230526", "save_truth": False},
    {"aoi_slug": "hankley-common", "s1_date": "20240520",
     "out_label": "hankley-common_20240520", "save_truth": True},
]


def _parse_target(spec: str) -> dict:
    """Parse ``aoi_slug:date[:label]`` into a target dict.

    Examples:
        brentmoor-area-training:20221008
        brentmoor-area-training:20221008:postfire
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"--target must be aoi_slug:YYYYMMDD[:label], got {spec!r}"
        )
    aoi_slug, s1_date = parts[0], parts[1]
    if len(s1_date) != 8 or not s1_date.isdigit():
        raise argparse.ArgumentTypeError(
            f"date must be YYYYMMDD (e.g. 20221008), got {s1_date!r}"
        )
    label_suffix = ("_" + parts[2]) if len(parts) == 3 else ""
    out_label = f"{aoi_slug}_{s1_date}{label_suffix}"
    return {"aoi_slug": aoi_slug, "s1_date": s1_date,
            "out_label": out_label, "save_truth": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", action="append", type=_parse_target, default=None,
                   metavar="AOI_SLUG:YYYYMMDD[:LABEL]",
                   help="Inference target. Repeat to predict multiple. "
                        "Default: project's three example sites.")
    p.add_argument("--run-name", default="v2_equivalent_initial",
                   help="Training run name; used to locate the .keras "
                        "checkpoint and to namespace outputs (default: %(default)s).")
    p.add_argument("--output-subdir", default=None,
                   help="Optional sub-directory under predictions/{unet,linear_baseline}/. "
                        "E.g. 'post_fire_2022' writes to predictions/unet/post_fire_2022/.")
    p.add_argument("--project", default=os.environ.get("GEE_S1S2_PROJECT_ID", "wildfire-495012"))
    p.add_argument("--bucket", default=os.environ.get("GEE_S1S2_BUCKET", "marcus-heath-fire-mapping"))
    p.add_argument("--prefix", default=os.environ.get("GEE_S1S2_PREFIX", "gee_s1s2_translator/operational_v1"))
    p.add_argument("--location", default=os.environ.get("GEE_S1S2_LOCATION", "europe-west2"))
    p.add_argument("--service-account",
                   default="gee-harvester@wildfire-495012.iam.gserviceaccount.com")
    p.add_argument("--package-version", default="0.5.1",
                   help="Training sdist version under "
                        "gs://<bucket>/<prefix>/vertex/packages/.")
    args = p.parse_args()

    targets = args.target or DEFAULT_TARGETS

    # Defer the heavy import so --help works without aiplatform installed.
    from google.cloud import aiplatform

    package_uri = (
        f"gs://{args.bucket}/{args.prefix}/vertex/packages/"
        f"gee_s1s2_translator_training-{args.package_version}.tar.gz"
    )
    aiplatform.init(
        project=args.project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}/{args.prefix}/vertex/staging",
    )

    env = [
        {"name": "GEE_S1S2_PROJECT_ID", "value": args.project},
        {"name": "GEE_S1S2_BUCKET", "value": args.bucket},
        {"name": "GEE_S1S2_PREFIX", "value": args.prefix},
        {"name": "GEE_S1S2_LOCATION", "value": args.location},
        {"name": "GEE_S1S2_TRAINING_RUN_NAME", "value": args.run_name},
        {"name": "GEE_S1S2_PREDICT_TARGETS_JSON", "value": json.dumps(targets)},
    ]
    if args.output_subdir:
        env.append({"name": "GEE_S1S2_PREDICT_OUTPUT_SUBDIR",
                    "value": args.output_subdir})

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri":
                    "europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest",
                "package_uris": [package_uri],
                "python_module": "training.predict_aois",
                "env": env,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=f"predict-aois-{args.run_name}",
        worker_pool_specs=worker_pool_specs,
    )

    print(f"Submitting inference over {len(targets)} target(s) to {args.location}...")
    for t in targets:
        print(f"  {t['out_label']} (aoi={t['aoi_slug']}, s1_date={t['s1_date']})")
    job.submit(service_account=args.service_account)
    print(f"\nJob ID:  {job.name}")
    print(f"State:   {int(job.state)}")
    print(f"Console: https://console.cloud.google.com/vertex-ai/"
          f"locations/{args.location}/training/{job.name}?project={args.project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
