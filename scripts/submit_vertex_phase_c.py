"""Submit the Phase C Vertex Custom Training job.

Phase C = architectural residual experiment versus the v2-equivalent
baseline. The U-Net learns a delta on top of a 1x1-conv linear baseline
of the S1 input; the final output is sigmoid(baseline_logits +
delta_logits). All other hyperparameters identical to the baseline.

See ``training/src/training/train_unet.py`` for the dispatch logic.
"""
from google.cloud import aiplatform

PROJECT_ID = "wildfire-495012"
LOCATION = "europe-west2"
GCS_BUCKET = "marcus-heath-fire-mapping"
GCS_PREFIX = "gee_s1s2_translator/operational_v1"
TRAINING_PACKAGE_VERSION = "0.5.0"
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
            "python_module": "training.train_unet",
            "env": [
                {"name": "GEE_S1S2_PROJECT_ID", "value": PROJECT_ID},
                {"name": "GEE_S1S2_BUCKET", "value": GCS_BUCKET},
                {"name": "GEE_S1S2_PREFIX", "value": GCS_PREFIX},
                {"name": "GEE_S1S2_LOCATION", "value": LOCATION},
                {"name": "GEE_S1S2_TRAINING_RUN_NAME", "value": "phase_c_residual"},
                {"name": "GEE_S1S2_PHASE", "value": "c"},
                # All other hyperparameters fall through to the baseline
                # defaults (max_epochs=80, patience=15, lr=1e-4).
            ],
        },
    }
]

job = aiplatform.CustomJob(
    display_name="phase-c-residual",
    worker_pool_specs=worker_pool_specs,
)

print(f"Submitting Phase C training to {LOCATION} as {SERVICE_ACCOUNT}...")
job.submit(service_account=SERVICE_ACCOUNT)
print(f"Job ID:  {job.name}")
print(f"State:   {int(job.state)}")
print(f"Console: https://console.cloud.google.com/vertex-ai/locations/{LOCATION}/training/{job.name}?project={PROJECT_ID}")
