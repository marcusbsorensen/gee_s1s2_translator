"""Minimal setup.py so the training source can be packaged as an sdist
for Vertex AI Custom Training jobs. Vertex's CustomPythonPackageTrainingJob
expects a tarball it can pip-install on the worker node.

Pinned dependencies are intentionally empty: the Vertex pre-built TF GPU
container already provides tensorflow + numpy at compatible versions, so
we only need to expose the ``training`` package itself.
"""
from setuptools import setup, find_packages

setup(
    name="gee_s1s2_translator_training",
    version="0.9.1",
    description="Phase 2/3 U-Net training + inference (incl. cosine-blended mosaic + post-fit affine calibration + PNG previews) + endpoint deployment for the GEE S1->S2 translator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        # Ship the post-fit affine calibration JSON so predict_aois can
        # apply it at inference without an external download.
        "training": ["calibration/*.json"],
    },
    include_package_data=True,
    python_requires=">=3.10",
    # rasterio is needed by predict_aois for GeoTIFF output; pre-built TF image
    # ships numpy + tensorflow but not rasterio + matplotlib.
    # google-cloud-aiplatform is needed by deploy_endpoint for the Vertex AI
    # Endpoint lifecycle. matplotlib is needed by the preview-PNG renderer.
    install_requires=[
        "rasterio>=1.3",
        "matplotlib>=3.5",
        "google-cloud-aiplatform>=1.50",
    ],
)
