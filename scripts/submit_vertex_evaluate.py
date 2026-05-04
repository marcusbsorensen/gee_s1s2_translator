"""Submit one Vertex Custom Job per training run to evaluate it.

Loads the .keras checkpoint, runs the test split through the model,
computes per-band MAE/RMSE + the variance-collapse diagnostic, writes
``validation_report.md`` + ``validation_metrics.json`` alongside the
checkpoint. CPU-only is enough — inference + numpy on ~377 patches is
trivial — so each job runs on n1-standard-4 + 1× T4 (the tf-gpu image
requires an accelerator) for ~5 min.

Targets are passed as a list so we evaluate the Phase B + Phase C
runs in parallel, each in its own job.
"""
from google.cloud import aiplatform

PROJECT_ID = "wildfire-495012"
LOCATION = "europe-west2"
GCS_BUCKET = "marcus-heath-fire-mapping"
GCS_PREFIX = "gee_s1s2_translator/operational_v1"
TRAINING_PACKAGE_VERSION = "0.5.1"
PACKAGE_URI = (
    f"gs://{GCS_BUCKET}/{GCS_PREFIX}/vertex/packages/"
    f"gee_s1s2_translator_training-{TRAINING_PACKAGE_VERSION}.tar.gz"
)
EXECUTOR_IMAGE = "europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest"
SERVICE_ACCOUNT = "gee-harvester@wildfire-495012.iam.gserviceaccount.com"

aiplatform.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=f"gs://{GCS_BUCKET}/{GCS_PREFIX}/vertex/staging",
)

TARGETS = [
    {"run_name": "phase_b_loss_activation", "phase": "b"},
    {"run_name": "phase_c_residual",       "phase": "c"},
]

submitted: list[tuple[str, str]] = []
for tgt in TARGETS:
    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": EXECUTOR_IMAGE,
                "package_uris": [PACKAGE_URI],
                "python_module": "training.evaluate_run",
                "env": [
                    {"name": "GEE_S1S2_PROJECT_ID", "value": PROJECT_ID},
                    {"name": "GEE_S1S2_BUCKET", "value": GCS_BUCKET},
                    {"name": "GEE_S1S2_PREFIX", "value": GCS_PREFIX},
                    {"name": "GEE_S1S2_LOCATION", "value": LOCATION},
                    {"name": "GEE_S1S2_TRAINING_RUN_NAME", "value": tgt["run_name"]},
                    {"name": "GEE_S1S2_PHASE", "value": tgt["phase"]},
                ],
            },
        }
    ]
    job = aiplatform.CustomJob(
        display_name=f"evaluate-{tgt['run_name'].replace('_', '-')}",
        worker_pool_specs=worker_pool_specs,
    )
    print(f"Submitting eval for {tgt['run_name']} ({tgt['phase']}) ...")
    job.submit(service_account=SERVICE_ACCOUNT)
    submitted.append((tgt["run_name"], job.name))
    print(f"  Job ID: {job.name}")

print("\nAll evaluation jobs submitted:")
for run_name, jid in submitted:
    print(f"  {run_name}: {jid}")
    print(f"    Console: https://console.cloud.google.com/vertex-ai/locations/{LOCATION}/training/{jid}?project={PROJECT_ID}")
