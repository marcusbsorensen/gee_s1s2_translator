"""Vertex AI Endpoint deployment + test + teardown for the trained U-Net.

Designed to run as a single Vertex Custom Job so the entire deploy → test →
teardown lifecycle is captured in one set of logs and one cost line. The
job:

1. Downloads the .keras checkpoint from GCS, rebuilds the architecture,
   and loads the weights (same load_weights pattern the trainer uses).
2. Exports the model as a TF SavedModel and uploads it to GCS.
3. Calls the Vertex AI SDK to upload to Model Registry, create an
   Endpoint, and deploy the model with a CPU-only serving container.
4. Sends a single test prediction (one held-out S1 patch from the
   manifest) and compares the response against the offline forward pass
   numerically.
5. Tears down (undeploy → delete endpoint → optionally delete model).

Cost discipline: CPU-only n1-standard-2 serving is ~£0.06/h on Vertex.
A typical end-to-end run is 15-25 minutes (deploy is the slow step), so
the marginal cost of the test cycle is ~£0.02-0.03. The hard cap is
enforced by always tearing down at the end.

Configuration via env vars:

  GEE_S1S2_PROJECT_ID
  GEE_S1S2_BUCKET
  GEE_S1S2_PREFIX
  GEE_S1S2_TRAINING_RUN_NAME (default: v2_equivalent_initial)
  GEE_S1S2_KEEP_MODEL=true to keep the Model Registry entry after teardown
    (default: deleted along with the endpoint, since a redeploy can rebuild it)
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import time
from typing import List

import numpy as np
import tensorflow as tf

from .data import S1_BANDS, S2_BANDS, S1Stats, _band_feature_spec
from .model import build_unet

LOG = logging.getLogger("deploy_endpoint")


def _env(name: str, default=None):
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"Required env var {name} is not set.")
    return val


def _load_unet_weights_from_gcs(uri: str) -> tf.keras.Model:
    """Download the .keras checkpoint, build the architecture, load weights."""
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        local = tmp.name
    try:
        tf.io.gfile.copy(uri, local, overwrite=True)
        unet = build_unet(input_shape=(256, 256, 2), out_channels=6, base_channels=32)
        unet.load_weights(local)
        return unet
    finally:
        try:
            os.remove(local)
        except OSError:
            pass


def _export_savedmodel(model: tf.keras.Model, gcs_uri: str) -> None:
    """Export Keras model as TF SavedModel + upload to GCS.

    Vertex's prediction containers expect SavedModel; the .keras zip
    archive is a Keras-only format and isn't directly servable.
    """
    with tempfile.TemporaryDirectory() as td:
        local_export = os.path.join(td, "saved_model")
        model.export(local_export)  # Keras 3: writes SavedModel via tf.saved_model.save
        # Upload directory contents to GCS
        for root, _dirs, files in os.walk(local_export):
            for f in files:
                local_path = os.path.join(root, f)
                rel = os.path.relpath(local_path, local_export)
                # Normalise separators for gs://
                rel_url = rel.replace(os.sep, "/")
                target = gcs_uri.rstrip("/") + "/" + rel_url
                tf.io.gfile.copy(local_path, target, overwrite=True)
                LOG.info("uploaded %s", target)


def _load_test_patch(s1_stats: S1Stats, manifest_uri: str) -> np.ndarray:
    """Pull the FIRST test-split patch from the manifest, return normalised S1."""
    import csv
    with tf.io.gfile.GFile(manifest_uri, "r") as f:
        rows = list(csv.DictReader(f))
    test_rows = [r for r in rows if r["split"] == "test"]
    if not test_rows:
        raise RuntimeError("No test-split rows in manifest")
    uri = test_rows[0]["tfrecord_uri"].replace("*.tfrecord.gz", ".tfrecord.gz")
    LOG.info("Test patch: %s", uri)
    spec = _band_feature_spec(256)
    raw = next(iter(tf.data.TFRecordDataset([uri], compression_type="GZIP")))
    parsed = tf.io.parse_single_example(raw, spec)
    s1 = tf.stack([tf.reshape(parsed[b], [256, 256]) for b in S1_BANDS], axis=-1)
    s1 = tf.where(tf.math.is_finite(s1), s1, tf.zeros_like(s1)).numpy()
    mean = np.array([s1_stats.mean[b] for b in S1_BANDS], dtype=np.float32)
    std = np.array([s1_stats.std[b] for b in S1_BANDS], dtype=np.float32)
    return ((s1 - mean) / std).astype(np.float32)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    project_id = _env("GEE_S1S2_PROJECT_ID")
    bucket = _env("GEE_S1S2_BUCKET")
    prefix = os.environ.get("GEE_S1S2_PREFIX", "gee_s1s2_translator/operational_v1")
    run_name = os.environ.get("GEE_S1S2_TRAINING_RUN_NAME", "v2_equivalent_initial")
    location = os.environ.get("GEE_S1S2_LOCATION", "europe-west2")
    keep_model = os.environ.get("GEE_S1S2_KEEP_MODEL", "false").lower() in {"true", "1", "yes"}

    base = f"gs://{bucket}/{prefix}"
    keras_uri = f"{base}/models/{run_name}/unet.keras"
    s1_stats_uri = f"{base}/s1_stats.json"
    manifest_uri = f"{base}/manifest.csv"
    saved_model_uri = f"{base}/models/{run_name}/saved_model"

    LOG.info("project=%s bucket=%s run=%s location=%s",
             project_id, bucket, run_name, location)

    # 1) Load .keras + 2) Export as SavedModel + upload to GCS
    LOG.info("Loading U-Net weights from %s ...", keras_uri)
    unet = _load_unet_weights_from_gcs(keras_uri)
    LOG.info("Exporting SavedModel + uploading to %s ...", saved_model_uri)
    _export_savedmodel(unet, saved_model_uri)

    # 3) Upload to Model Registry + create + deploy Endpoint
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=location,
                    staging_bucket=f"{base}/vertex/staging")

    LOG.info("Uploading SavedModel to Vertex Model Registry ...")
    model = aiplatform.Model.upload(
        display_name=f"unet-s1-to-s2-{run_name}",
        artifact_uri=saved_model_uri,
        # CPU TF prediction container.
        serving_container_image_uri=(
            "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest"
        ),
        sync=True,
    )
    LOG.info("Model uploaded. Resource: %s", model.resource_name)

    LOG.info("Creating Endpoint ...")
    endpoint = aiplatform.Endpoint.create(
        display_name=f"unet-s1-to-s2-{run_name}-endpoint",
        sync=True,
    )
    LOG.info("Endpoint created. Resource: %s", endpoint.resource_name)

    deploy_start = time.time()
    LOG.info("Deploying model to endpoint (this is the slow step, ~5-15 min) ...")
    deployed = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="unet-s1-to-s2-deployed",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_percentage=100,
        sync=True,
    )
    deploy_secs = time.time() - deploy_start
    LOG.info("Deployment complete in %.0f s", deploy_secs)

    # 4) Test prediction + numerical comparison
    try:
        LOG.info("Loading test patch ...")
        with tf.io.gfile.GFile(s1_stats_uri, "r") as fh:
            sd = json.load(fh)
        stats = S1Stats(mean=sd["mean"], std=sd["std"])
        test_s1 = _load_test_patch(stats, manifest_uri)
        LOG.info("Test S1 patch shape: %s, dtype: %s", test_s1.shape, test_s1.dtype)

        # Endpoint expects instances list. SavedModel's signature is
        # input "input_layer" / "input_1" / similar; the prediction
        # container handles serialisation. Send as a 4-d tensor (batch=1).
        instances = [test_s1.tolist()]
        LOG.info("Calling endpoint.predict() ...")
        t0 = time.time()
        response = endpoint.predict(instances=instances)
        endpoint_latency_s = time.time() - t0
        LOG.info("Endpoint predicted in %.2f s", endpoint_latency_s)

        endpoint_pred = np.array(response.predictions[0], dtype=np.float32)
        LOG.info("Endpoint pred shape: %s", endpoint_pred.shape)

        # Offline forward pass
        LOG.info("Running offline forward pass for parity check ...")
        offline_pred = unet.predict(test_s1[None], verbose=0)[0]
        max_abs_diff = float(np.abs(endpoint_pred - offline_pred).max())
        mean_abs_diff = float(np.abs(endpoint_pred - offline_pred).mean())
        LOG.info("Parity: max |endpoint - offline| = %.6f, mean = %.6f",
                 max_abs_diff, mean_abs_diff)

        # Numerical tolerance: float32 inference noise ≤ 1e-5
        tol = 1e-5
        passed = max_abs_diff < tol
        LOG.info("Parity check: %s (tolerance=%g)", "PASS" if passed else "FAIL", tol)

        # Persist a small JSON record so the result is verifiable post-teardown.
        record = {
            "test_passed": passed,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "tolerance": tol,
            "endpoint_pred_shape": list(endpoint_pred.shape),
            "endpoint_latency_seconds": endpoint_latency_s,
            "deploy_seconds": deploy_secs,
            "machine_type": "n1-standard-2",
            "container_image": "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
            "model_resource": model.resource_name,
            "endpoint_resource": endpoint.resource_name,
        }
        out_uri = f"{base}/models/{run_name}/endpoint_deployment_test.json"
        with tf.io.gfile.GFile(out_uri, "w") as fh:
            json.dump(record, fh, indent=2)
        LOG.info("Wrote test record: %s", out_uri)

    finally:
        # 5) ALWAYS tear down — endpoint billing accrues per-hour while deployed.
        LOG.info("Undeploying model from endpoint ...")
        endpoint.undeploy_all(sync=True)
        LOG.info("Deleting endpoint ...")
        endpoint.delete(sync=True)
        LOG.info("Endpoint deleted; no further compute billing.")
        if not keep_model:
            LOG.info("Deleting model from Registry (set GEE_S1S2_KEEP_MODEL=true to keep) ...")
            model.delete(sync=True)
            LOG.info("Model deleted.")
        else:
            LOG.info("Keeping model in Registry: %s", model.resource_name)

    LOG.info("Endpoint lifecycle complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
