"""Earth Engine and Google Cloud Storage authentication helpers.

Two authentication paths are supported:

1. **Service account** (recommended for headless runs and operational
   deployment). The path to the service account JSON is read from
   ``GOOGLE_APPLICATION_CREDENTIALS``. The same key authorises both Earth
   Engine and Cloud Storage when the service account has the relevant IAM
   roles (Earth Engine Resource Viewer, Storage Object Admin on the bucket).
2. **Interactive user credentials** (suitable for local development). Falls
   back to whatever ``earthengine authenticate`` and
   ``gcloud auth application-default login`` have stored.

Both paths fail loudly with a clear error message pointing at the README's
setup section, so a bad config never silently produces wrong results.

Sonia (operational reader): see ``docs/architecture_overview.md`` and the
README for which path to use under what circumstances.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

LOG = logging.getLogger(__name__)


class GeeAuthError(RuntimeError):
    """Raised when Earth Engine authentication fails for any reason."""


class GcsAuthError(RuntimeError):
    """Raised when Google Cloud Storage authentication fails for any reason."""


def load_env(env_file: Path | None = None) -> None:
    """Load ``.env`` into ``os.environ``. No-op if the file is missing."""
    if env_file is None:
        env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file, override=False)
        LOG.debug("Loaded environment from %s", env_file)


def get_gee_project_id() -> str:
    """Return ``GEE_PROJECT_ID`` from env, or raise with a helpful message."""
    pid = os.environ.get("GEE_PROJECT_ID", "").strip()
    if not pid:
        raise GeeAuthError(
            "GEE_PROJECT_ID is not set. Edit .env (copy from .env.example) and "
            "set GEE_PROJECT_ID to your Earth Engine project ID. If you have "
            "not registered for Earth Engine yet, see the README's "
            "'Setup: Earth Engine' section."
        )
    return pid


def get_gcs_bucket() -> str:
    bucket = os.environ.get("GCS_BUCKET", "").strip()
    if not bucket:
        raise GcsAuthError(
            "GCS_BUCKET is not set. Edit .env and set GCS_BUCKET to the name "
            "of a Google Cloud Storage bucket you control. If you have not "
            "created one, see the README's 'Setup: GCS bucket' section."
        )
    if bucket.startswith("gs://"):
        bucket = bucket[len("gs://") :]
    return bucket.rstrip("/")


def get_gcs_prefix() -> str:
    return os.environ.get("GCS_PREFIX", "gee_s1s2_translator").strip().strip("/")


def get_service_account_path() -> Path | None:
    """Return the resolved service account JSON path, or ``None``."""
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.exists():
        raise GeeAuthError(
            f"GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {p}. "
            "Either fix the path, or unset it to fall back to interactive "
            "credentials (after running 'earthengine authenticate')."
        )
    return p


def init_earthengine() -> None:
    """Authenticate and initialise Earth Engine with the configured project.

    Always called once at startup. Re-initialising in the same process is
    a no-op for ee.Initialize.
    """
    import ee  # heavy import; keep local

    project = get_gee_project_id()
    sa_path = get_service_account_path()
    try:
        if sa_path is not None:
            with sa_path.open("r", encoding="utf-8") as fh:
                import json
                sa_email = json.load(fh).get("client_email")
            if not sa_email:
                raise GeeAuthError(
                    f"Service account JSON {sa_path} has no 'client_email' field."
                )
            credentials = ee.ServiceAccountCredentials(sa_email, str(sa_path))
            ee.Initialize(credentials, project=project)
            LOG.info("Earth Engine initialised with service account %s (project=%s)",
                     sa_email, project)
        else:
            ee.Initialize(project=project)
            LOG.info("Earth Engine initialised with stored user credentials "
                     "(project=%s)", project)
    except Exception as exc:  # noqa: BLE001 - re-wrap with a useful message
        raise GeeAuthError(
            f"Earth Engine initialisation failed: {exc}. "
            "Run 'earthengine authenticate' for interactive auth, or set "
            "GOOGLE_APPLICATION_CREDENTIALS to a service account JSON. "
            "If you have not registered with Earth Engine, sign up at "
            "https://code.earthengine.google.com first."
        ) from exc


def get_gcs_client():
    """Return an authenticated google-cloud-storage Client."""
    from google.cloud import storage  # heavy import; keep local
    sa_path = get_service_account_path()
    try:
        if sa_path is not None:
            return storage.Client.from_service_account_json(str(sa_path))
        # Application default credentials
        return storage.Client()
    except Exception as exc:  # noqa: BLE001
        raise GcsAuthError(
            f"GCS client construction failed: {exc}. "
            "Run 'gcloud auth application-default login' for interactive "
            "credentials, or set GOOGLE_APPLICATION_CREDENTIALS to a "
            "service account JSON. The README documents both paths."
        ) from exc


def check_gcs_bucket(client=None) -> None:
    """Confirm the configured GCS bucket exists and the caller can read it."""
    bucket_name = get_gcs_bucket()
    if client is None:
        client = get_gcs_client()
    try:
        bucket = client.bucket(bucket_name)
        # bucket.exists() does a HEAD against the bucket; fast and cheap.
        if not bucket.exists():
            raise GcsAuthError(
                f"GCS bucket gs://{bucket_name}/ does not exist or is not "
                "visible to the current credentials. Either create it "
                "(see README 'Setup: GCS bucket') or check IAM."
            )
    except GcsAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GcsAuthError(
            f"GCS bucket check failed for gs://{bucket_name}/: {exc}"
        ) from exc
    LOG.info("GCS bucket gs://%s/ is reachable.", bucket_name)
