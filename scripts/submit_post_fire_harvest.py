"""Submit a Vertex Custom Job that harvests Sentinel-1/Sentinel-2 patches
for one or more AOIs over a single date window.

This is the operator-facing wrapper around the harvest entry point in
``gee_s1s2.run_post_fire_harvest``. It defaults to the project's
canonical post-fire 2022 cloud-penetration target (Brentmoor +
Poors Allotment, window ``post-fire 2022``); pass ``--aoi`` and
``--window`` to harvest a different combination.

Cost: ~£0.30 wall-clock for a typical 2-AOI harvest (most of that is
GEE export queue time, not Vertex compute). CPU-only n1-standard-4.

Examples
--------

# Default — re-harvest the post-fire 2022 cloud-covered window
python scripts/submit_post_fire_harvest.py

# Two AOIs over a different window
python scripts/submit_post_fire_harvest.py \\
    --aoi "Brentmoor area training" \\
    --aoi "Poors Allotment area training" \\
    --window "comparison 2024"
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--aoi", action="append", default=None,
                   metavar="AOI_NAME",
                   help='AOI display name (must match config/operational_v1.yaml). '
                        'Repeat to harvest multiple. Default: Brentmoor + Poors Allotment.')
    p.add_argument("--window", default=None,
                   help='Date-window name from config/operational_v1.yaml. '
                        'Default: "post-fire 2022".')
    p.add_argument("--project", default=os.environ.get("GEE_S1S2_PROJECT_ID", "wildfire-495012"))
    p.add_argument("--bucket", default=os.environ.get("GEE_S1S2_BUCKET", "marcus-heath-fire-mapping"))
    p.add_argument("--prefix", default=os.environ.get("GEE_S1S2_PREFIX", "gee_s1s2_translator/operational_v1"))
    p.add_argument("--location", default=os.environ.get("GEE_S1S2_LOCATION", "europe-west2"))
    p.add_argument("--service-account",
                   default="gee-harvester@wildfire-495012.iam.gserviceaccount.com")
    p.add_argument("--package-version", default="1.0.1",
                   help="Harvest sdist version under "
                        "gs://<bucket>/<prefix>/vertex/packages/.")
    args = p.parse_args()

    from google.cloud import aiplatform

    package_uri = (
        f"gs://{args.bucket}/{args.prefix}/vertex/packages/"
        f"gee_s1s2_harvest-{args.package_version}.tar.gz"
    )
    aiplatform.init(
        project=args.project,
        location=args.location,
        staging_bucket=f"gs://{args.bucket}/{args.prefix}/vertex/staging",
    )

    env = [
        {"name": "GEE_PROJECT_ID", "value": args.project},
        {"name": "GCS_BUCKET", "value": args.bucket},
        {"name": "GCS_PREFIX", "value": args.prefix},
    ]
    if args.aoi:
        env.append({"name": "GEE_S1S2_TARGET_AOIS",
                    "value": ",".join(args.aoi)})
    if args.window:
        env.append({"name": "GEE_S1S2_TARGET_WINDOW",
                    "value": args.window})

    worker_pool_specs = [
        {
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri":
                    "europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-15.py310:latest",
                "package_uris": [package_uri],
                "python_module": "gee_s1s2.run_post_fire_harvest",
                "env": env,
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name="harvest-post-fire",
        worker_pool_specs=worker_pool_specs,
    )

    aoi_label = ", ".join(args.aoi) if args.aoi else "(defaults)"
    print(f"Submitting harvest to {args.location} as {args.service_account}...")
    print(f"  AOIs:   {aoi_label}")
    print(f"  Window: {args.window or '(default: post-fire 2022)'}")
    job.submit(service_account=args.service_account)
    print(f"\nJob ID:  {job.name}")
    print(f"State:   {int(job.state)}")
    print(f"Console: https://console.cloud.google.com/vertex-ai/"
          f"locations/{args.location}/training/{job.name}?project={args.project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
