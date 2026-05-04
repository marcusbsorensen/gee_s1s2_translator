"""Submit a Vertex Custom Job that trains the v2-equivalent baseline U-Net
from scratch on the current harvest manifest.

This is the operator-facing path for re-training (e.g. after harvesting
a new region). For the experimental Phase B / Phase C variants, use the
dedicated ``submit_vertex_phase_b.py`` / ``submit_vertex_phase_c.py``
scripts instead — those set the variance-loss / residual-architecture
flags this script intentionally leaves alone.

Cost: ~£0.30–0.60 wall-clock on n1-standard-4 + 1× T4. Typical run is
20–30 minutes; max 80 epochs with patience=15 early-stopping.

Examples
--------

# Train with default run name (v2_equivalent_initial)
python scripts/submit_vertex_training.py

# Train and write artefacts under a custom run name
python scripts/submit_vertex_training.py --run-name my_first_retrain

# Dial in a specific budget (e.g. faster iteration during config testing)
python scripts/submit_vertex_training.py --max-epochs 40 --patience 10
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-name", default="v2_equivalent_initial",
                   help="Training run name; namespaces all outputs under "
                        "gs://<bucket>/<prefix>/models/<run_name>/.")
    p.add_argument("--max-epochs", type=int, default=80,
                   help="Maximum training epochs (default: %(default)s).")
    p.add_argument("--patience", type=int, default=15,
                   help="Early-stopping patience on val_rmse (default: %(default)s).")
    p.add_argument("--learning-rate", default="1e-4",
                   help="Initial learning rate (default: %(default)s).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-step batch size (default: %(default)s).")
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
                "python_module": "training.train_unet",
                "env": [
                    {"name": "GEE_S1S2_PROJECT_ID", "value": args.project},
                    {"name": "GEE_S1S2_BUCKET", "value": args.bucket},
                    {"name": "GEE_S1S2_PREFIX", "value": args.prefix},
                    {"name": "GEE_S1S2_LOCATION", "value": args.location},
                    {"name": "GEE_S1S2_TRAINING_RUN_NAME", "value": args.run_name},
                    {"name": "GEE_S1S2_MAX_EPOCHS", "value": str(args.max_epochs)},
                    {"name": "GEE_S1S2_EARLY_STOPPING_PATIENCE", "value": str(args.patience)},
                    {"name": "GEE_S1S2_LEARNING_RATE", "value": str(args.learning_rate)},
                    {"name": "GEE_S1S2_BATCH_SIZE", "value": str(args.batch_size)},
                    # GEE_S1S2_PHASE intentionally unset → baseline path.
                ],
            },
        }
    ]

    job = aiplatform.CustomJob(
        display_name=f"train-{args.run_name.replace('_', '-')}",
        worker_pool_specs=worker_pool_specs,
    )

    print(f"Submitting baseline training to {args.location} as {args.service_account}...")
    print(f"  run_name:   {args.run_name}")
    print(f"  max_epochs: {args.max_epochs}, patience: {args.patience}, "
          f"lr: {args.learning_rate}, batch: {args.batch_size}")
    job.submit(service_account=args.service_account)
    print(f"\nJob ID:  {job.name}")
    print(f"State:   {int(job.state)}")
    print(f"Console: https://console.cloud.google.com/vertex-ai/"
          f"locations/{args.location}/training/{job.name}?project={args.project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
