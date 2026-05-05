"""Command line interface for the GEE harvesting pipeline.

Five commands:

* ``init``           validate config + check GEE auth + check GCS bucket
* ``calibrate_check`` run the S1 calibration validation against MPC RTC
* ``harvest --dry-run`` candidate counts per (AOI, window), no exports
* ``harvest``        run the full harvest, exporting to GCS
* ``manifest summary`` print manifest summary (matches v2 format)

Each command writes a timestamped log to ``./local_cache/logs/`` so the
operational user (Sonia) can inspect what each run did. Logs are tiny
plaintext; nothing methodologically substantive lives here, the v2
manifest in GCS is the source of truth.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .auth import (
    GcsAuthError,
    GeeAuthError,
    check_gcs_bucket,
    get_gcs_bucket,
    get_gcs_client,
    get_gcs_prefix,
    get_gee_project_id,
    init_earthengine,
    load_env,
)
from .config import Config, load_config

app = typer.Typer(
    add_completion=False,
    help="GEE Sentinel-1/Sentinel-2 harvesting for lowland heath fire mapping.",
)
manifest_app = typer.Typer(help="Inspect the manifest in GCS.")
app.add_typer(manifest_app, name="manifest")

console = Console()


def _configure_logging(workspace: Path, command: str, verbose: bool) -> None:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    level = "DEBUG" if verbose else os.environ.get("GEE_S1S2_LOG_LEVEL", "INFO")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"{command}_{timestamp}.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _bootstrap(config_path: Path, command: str, verbose: bool) -> Config:
    load_env()
    config = load_config(config_path)
    config.project.workspace.mkdir(parents=True, exist_ok=True)
    _configure_logging(config.project.workspace, command, verbose)
    return config


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #

@app.command("init")
def cmd_init(
    config: Path = typer.Option(Path("config/operational_v1.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Validate config and confirm GEE + GCS access."""
    try:
        cfg = _bootstrap(config, "init", verbose)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Config invalid:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Config OK.[/green]  Project: {cfg.project.name}")
    console.print(f"  AOIs:       {len(cfg.aois)}  "
                  f"({sum(1 for a in cfg.aois if a.role != 'target')} training)")
    console.print(f"  Windows:    {len(cfg.date_windows)}")
    console.print(f"  Force-split: "
                  + ", ".join(f"{a.name} -> {a.force_split}"
                              for a in cfg.aois if a.force_split) or "(none)")

    # GEE
    try:
        init_earthengine()
        console.print(f"[green]GEE OK.[/green]  project={get_gee_project_id()}")
    except GeeAuthError as exc:
        console.print(f"[red]GEE auth failed:[/red] {exc}")
        raise typer.Exit(code=2)

    # GCS
    try:
        client = get_gcs_client()
        check_gcs_bucket(client)
        console.print(
            f"[green]GCS OK.[/green]  "
            f"gs://{get_gcs_bucket()}/{get_gcs_prefix()}/operational_v1/"
        )
    except GcsAuthError as exc:
        console.print(f"[red]GCS auth/bucket failed:[/red] {exc}")
        raise typer.Exit(code=3)


# --------------------------------------------------------------------------- #
# calibrate_check
# --------------------------------------------------------------------------- #

@app.command("calibrate_check")
def cmd_calibrate_check(
    config: Path = typer.Option(Path("config/operational_v1.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose"),
    n_samples: int = typer.Option(3, help="How many (AOI, date) samples to compare."),
) -> None:
    """Validate the GEE calibration pipeline against MPC RTC reference samples.

    Picks ``n_samples`` (AOI, date) combinations from v2's manifest where MPC
    RTC has matching acquisitions, harvests both, and reports per-pixel mean
    and std of the dB difference. Pass criterion: mean offset within
    ±2 dB and relative std difference within 30% on the AOI mask.
    Writes the validation result to ``docs/calibration_methodology.md``.

    This command requires:
    * GEE auth (init must pass first)
    * Access to the v2 archive at ../s1s2-translator/data/runs/v2_diverse_heath/
      so we can read the manifest's S1 ids for the comparison.
    """
    cfg = _bootstrap(config, "calibrate_check", verbose)

    try:
        init_earthengine()
    except GeeAuthError as exc:
        console.print(f"[red]GEE auth failed:[/red] {exc}")
        raise typer.Exit(code=2)

    # Implementation lives in a separate module rather than inline so it can
    # be run from a notebook. Keeping the CLI thin.
    from . import validation

    rows = validation.run_calibration_validation(cfg, n_samples=n_samples)
    table = Table(title="S1 calibration: GEE vs MPC RTC")
    table.add_column("AOI"); table.add_column("date")
    table.add_column("VV mean Δ (dB)", justify="right")
    table.add_column("VH mean Δ (dB)", justify="right")
    table.add_column("VV std ratio", justify="right")
    table.add_column("VH std ratio", justify="right")
    table.add_column("verdict")
    for r in rows:
        table.add_row(
            r["aoi"], r["date"],
            f"{r['vv_mean_delta_db']:+.2f}",
            f"{r['vh_mean_delta_db']:+.2f}",
            f"{r['vv_std_ratio']:.2f}",
            f"{r['vh_std_ratio']:.2f}",
            r["verdict"],
        )
    console.print(table)

    # Write result into docs/calibration_methodology.md (append-only)
    validation.append_validation_to_doc(rows)


# --------------------------------------------------------------------------- #
# harvest
# --------------------------------------------------------------------------- #

@app.command("harvest")
def cmd_harvest(
    config: Path = typer.Option(Path("config/operational_v1.yaml"), "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Search and pair only; no exports."),
    aoi: str | None = typer.Option(
        None, "--aoi",
        help="Restrict harvest to one AOI. Used by Phase 1 review-gate small harvest.",
    ),
    window: str | None = typer.Option(
        None, "--window",
        help="Restrict harvest to one date window (used with --aoi).",
    ),
    include_inference: bool = typer.Option(
        False, "--include-inference",
        help="Also harvest windows with role: inference (e.g. post-fire 2022, "
             "early post-fire 2022). Off by default to keep training-window "
             "harvests focused.",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the harvest. Exports to GCS in normal mode, no exports in dry-run."""
    cfg = _bootstrap(config, "harvest", verbose)
    try:
        init_earthengine()
    except GeeAuthError as exc:
        console.print(f"[red]GEE auth failed:[/red] {exc}")
        raise typer.Exit(code=2)
    if not dry_run:
        try:
            check_gcs_bucket()
        except GcsAuthError as exc:
            console.print(f"[red]GCS bucket check failed:[/red] {exc}")
            raise typer.Exit(code=3)

    from . import harvest

    summary = harvest.run_harvest(
        cfg, dry_run=dry_run, only_aoi=aoi, only_window=window,
        include_inference_windows=include_inference,
    )

    table = Table(title=("Harvest summary (dry-run)" if dry_run else "Harvest summary"))
    table.add_column("metric", style="bold"); table.add_column("count", justify="right")
    table.add_row("Candidate pairs (after temporal filter)", str(summary.candidates))
    table.add_row("New pairs (after dedup vs manifest)", str(summary.new_pairs))
    table.add_row("Accepted after AOI cloud check", str(summary.accepted_after_cloud))
    table.add_row("Written to manifest", str(summary.written_to_manifest))
    table.add_row("Patches written", str(summary.patches_written))
    console.print(table)

    if summary.per_bucket:
        bucket_table = Table(title="Candidate pairs per (AOI, window)")
        bucket_table.add_column("AOI"); bucket_table.add_column("window")
        bucket_table.add_column("candidates", justify="right")
        for (aoi_name, window_label), n in sorted(summary.per_bucket.items()):
            bucket_table.add_row(aoi_name, window_label, str(n))
        console.print(bucket_table)
        zero_aois = sorted(
            {aoi for (aoi, _w) in summary.per_bucket if not summary.per_bucket[(aoi, _w)]}
        )
        if zero_aois:
            console.print(
                f"[red]Zero-candidate AOIs:[/red] {', '.join(zero_aois)}"
            )

    if summary.skipped_excluded:
        skip_table = Table(title="Skipped (AOI, window) due to exclude_windows")
        skip_table.add_column("AOI"); skip_table.add_column("window")
        for aoi_name, window_label in summary.skipped_excluded:
            skip_table.add_row(aoi_name, window_label)
        console.print(skip_table)


# --------------------------------------------------------------------------- #
# manifest summary
# --------------------------------------------------------------------------- #

@manifest_app.command("summary")
def cmd_manifest_summary(
    config: Path = typer.Option(Path("config/operational_v1.yaml"), "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Print manifest summary in the same format as the v2 PyTorch project."""
    cfg = _bootstrap(config, "manifest_summary", verbose)
    try:
        client = get_gcs_client()
        check_gcs_bucket(client)
    except GcsAuthError as exc:
        console.print(f"[red]GCS access failed:[/red] {exc}")
        raise typer.Exit(code=3)

    from .manifest import GcsManifest
    blob_path = (
        f"{get_gcs_prefix()}/operational_v1/{cfg.storage.manifest_path}"
    ).replace("\\", "/")
    m = GcsManifest(client, get_gcs_bucket(), blob_path)
    s = m.summary()
    console.print(f"[bold]Total rows:[/bold] {s['total']}")
    for header, key in (("By AOI", "by_aoi"),
                        ("By window", "by_window"),
                        ("By split", "by_split")):
        console.print(f"\n  {header}:")
        for k, v in sorted(s.get(key, {}).items()):
            console.print(f"    {k or '(blank)'}: {v}")


def main() -> None:
    """Entry point for the ``gee_s1s2`` console script."""
    app()


if __name__ == "__main__":
    main()
