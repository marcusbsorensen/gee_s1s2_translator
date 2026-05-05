"""Submit Phase B v2 — variance-active corrected-hyperparameter re-run.

The original Phase B run early-stopped at epoch 50 with the best
checkpoint at epoch 25, which was BEFORE the variance term's warmup
epoch (30). The variance hypothesis therefore went untested. This v2
corrects the planning bug:

- variance warmup at epoch 15 (was 30)
- early-stopping patience 40 (was 25)

so that the variance term has at least ~25 epochs of active training
before patience could fire, while still letting the L1+L2 base loss
establish a sensible solution before variance modifies the landscape.

Output to gs://<bucket>/<prefix>/models/phase_b_v2_variance_active/.
All other Phase B modifications carry through unchanged: linear output
activation, cosine LR 1e-4 -> 1e-5, MAX_EPOCHS=120.
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
                {"name": "GEE_S1S2_TRAINING_RUN_NAME", "value": "phase_b_v2_variance_active"},
                {"name": "GEE_S1S2_PHASE", "value": "b"},
                {"name": "GEE_S1S2_MAX_EPOCHS", "value": "120"},
                {"name": "GEE_S1S2_EARLY_STOPPING_PATIENCE", "value": "40"},
                {"name": "GEE_S1S2_LEARNING_RATE", "value": "1e-4"},
                {"name": "GEE_S1S2_PHASE_B_VARIANCE_WEIGHT", "value": "0.3"},
                {"name": "GEE_S1S2_PHASE_B_VARIANCE_WARMUP_EPOCH", "value": "15"},
                {"name": "GEE_S1S2_PHASE_B_COSINE_MIN_LR", "value": "1e-5"},
            ],
        },
    }
]

job = aiplatform.CustomJob(
    display_name="phase-b-v2-variance-active",
    worker_pool_specs=worker_pool_specs,
)

print(f"Submitting Phase B v2 training to {LOCATION} as {SERVICE_ACCOUNT}...")
job.submit(service_account=SERVICE_ACCOUNT)
print(f"Job ID:  {job.name}")
print(f"State:   {int(job.state)}")
print(f"Console: https://console.cloud.google.com/vertex-ai/locations/{LOCATION}/training/{job.name}?project={PROJECT_ID}")
