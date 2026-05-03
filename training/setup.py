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
    version="0.3.0",
    description="Phase 2 U-Net training + inference package for the GEE S1->S2 translator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    # rasterio is needed by predict_aois for GeoTIFF output; pre-built TF image
    # ships numpy + tensorflow but not rasterio.
    install_requires=["rasterio>=1.3"],
)
