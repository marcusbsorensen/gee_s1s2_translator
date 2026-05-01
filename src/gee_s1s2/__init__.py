"""gee_s1s2_translator: GEE port of the v2 lowland heath fire mapping pipeline.

Phase 1 of a four-phase rebuild for operational deployment. This package
covers the harvesting layer (Phase 1): Sentinel-1 GRD calibration, Sentinel-2
SR_HARMONIZED filtering, paired patch sampling, and TFRecord export to
Google Cloud Storage. Training (Phase 2) runs in Colab; inference deployment
(Phase 3) targets Vertex AI; operational handover (Phase 4) packages the
full toolchain for Sonia's professional use.

The configuration surface mirrors the v2 PyTorch project's
``config/example_mpc_augmented_1.yaml`` byte-for-byte at the AOI and date
window level; only the data-source sections (sentinel1/sentinel2) and three
new blocks (calibration, export, training_split) differ. See
``docs/architecture_overview.md`` for the v2-to-v1-operational mapping.
"""

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
