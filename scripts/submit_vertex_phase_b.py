"""Submit the Phase B Vertex Custom Training job.

Phase B = loss + output activation experiment versus the v2-equivalent
baseline. See ``training/src/training/train_unet.py`` for the dispatch
logic; this script only sets the env vars and submits.

Differences from baseline:
- ``GEE_S1S2_PHASE=b``
- Linear output activation (sigmoid removed; clip to [0, 1] at inference)
- Combined loss = L1 + 0.5*L2 + 0.3*variance_term, variance term active
  from epoch 30 onward.
- Cosine LR decay from 1e-4 to 1e-5 over the full 120-epoch budget.
- MAX_EPOCHS=120, EARLY_STOPPING_PATIENCE=25.
- TRAINING_RUN_NAME=phase_b_loss_activation (separate output prefix).
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
                {"name": "GEE_S1S2_TRAINING_RUN_NAME", "value": "phase_b_loss_activation"},
                {"name": "GEE_S1S2_PHASE", "value": "b"},
                {"name": "GEE_S1S2_MAX_EPOCHS", "value": "120"},
                {"name": "GEE_S1S2_EARLY_STOPPING_PATIENCE", "value": "25"},
                {"name": "GEE_S1S2_LEARNING_RATE", "value": "1e-4"},
                {"name": "GEE_S1S2_PHASE_B_VARIANCE_WEIGHT", "value": "0.3"},
                {"name": "GEE_S1S2_PHASE_B_VARIANCE_WARMUP_EPOCH", "value": "30"},
                {"name": "GEE_S1S2_PHASE_B_COSINE_MIN_LR", "value": "1e-5"},
            ],
        },
    }
]

job = aiplatform.CustomJob(
    display_name="phase-b-loss-activation",
    worker_pool_specs=worker_pool_specs,
)

print(f"Submitting Phase B training to {LOCATION} as {SERVICE_ACCOUNT}...")
job.submit(service_account=SERVICE_ACCOUNT)
print(f"Job ID:  {job.name}")
print(f"State:   {int(job.state)}")
print(f"Console: https://console.cloud.google.com/vertex-ai/locations/{LOCATION}/training/{job.name}?project={PROJECT_ID}")
