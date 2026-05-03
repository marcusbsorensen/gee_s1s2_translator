"""Phase 2 training modules: U-Net + linear baseline for S1->S2 translation.

The data pipeline reads the Phase 1 GCS-harvested TFRecord shards
(one 256x256x8 patch per record), routes patches to train/val/test
according to the manifest's ``split`` column, and feeds them to the
U-Net (or linear baseline) for training. Validation includes per-band
MAE/RMSE and the variance-collapse diagnostic against patch-specific
truth std (the v2 refinement).

See ``training/README.md`` for the Colab walkthrough; see the parent
``docs/methodology_divergences.md`` for the GEE-vs-v2 differences that
matter when reading the comparison-vs-v2 cell of the notebook.
"""
__version__ = "0.1.0"
