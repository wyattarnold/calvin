"""
Entry point for the Calvin Network App.

Usage::

    # Explicit mode (original)
    python -m calvin.app serve \\
        --data /path/to/calvin-network-data/data \\
        --study my-models/calvin-pf \\
        --port 8000

    # Local mode: auto-discover studies under ./my-models/
    python -m calvin.app serve --data /path/to/calvin-network-data/data --local

    # Hosted mode: load network + studies from bundled data.zip
    python -m calvin.app serve --hosted

    # Bundle data.zip for hosted deployment
    python -m calvin.app bundle \\
        --data /path/to/calvin-network-data/data \\
        --study my-models/calvin-pf
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calvin.app",
        description="Calvin Network App — FastAPI web server for CALVIN visualization",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- serve ----
    p_serve = subparsers.add_parser("serve", help="Start the web server")
    p_serve.add_argument(
        "--data", "-d",
        default=None,
        help=(
            "Path to calvin-network-data/data directory. "
            "Required unless --hosted is set."
        ),
    )
    p_serve.add_argument(
        "--study", "-s",
        action="append",
        default=[],
        dest="studies",
        metavar="PATH",
        help=(
            "Path to a model run directory containing results/. "
            "Can be repeated to load multiple studies. "
            "Ignored when --local or --hosted is set."
        ),
    )
    p_serve.add_argument(
        "--default-study",
        default=None,
        help="Name of the study to make active by default (basename of --study path)",
    )

    mode_group = p_serve.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--local",
        action="store_true",
        help=(
            "Auto-discover studies under ./my-models/ (any subdir with a results/ folder). "
            "--data is still required; --study is ignored."
        ),
    )
    mode_group.add_argument(
        "--hosted",
        action="store_true",
        help=(
            "Load network + studies from bundled calvin/app/data.zip. "
            "Extracts to a temp directory at startup. --data and --study are ignored."
        ),
    )

    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development only)",
    )
    p_serve.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    # ---- bundle ----
    p_bundle = subparsers.add_parser(
        "bundle",
        help="Create data.zip for hosted deployment",
    )
    p_bundle.add_argument(
        "--data", "-d",
        required=True,
        help="Path to calvin-network-data/data directory (written to network/ in zip)",
    )
    p_bundle.add_argument(
        "--study", "-s",
        action="append",
        default=[],
        dest="studies",
        metavar="PATH",
        required=True,
        help=(
            "Path to a model run directory. results/*.csv are written to "
            "studies/{name}/results/ in the zip. Can be repeated."
        ),
    )
    p_bundle.add_argument(
        "--output", "-o",
        default=None,
        help="Output zip path (default: calvin/app/data.zip relative to this file)",
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_local_studies(base: Path) -> list[Path]:
    """Return subdirs of *base* that contain a results/ subdirectory."""
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / "results").is_dir())


def _extract_bundle() -> tuple[Path, list[Path]]:
    """Extract the bundled data.zip to a temp directory.

    Returns
    -------
    tmpdir
        The extracted temp directory root. Contains ``prebuilt/`` with
        pre-built network JSON files and ``studies/`` with result CSVs.
    study_paths
        One ``<tmpdir>/studies/<name>`` entry per study in the zip.
    """
    bundle = Path(__file__).parent / "data.zip"
    if not bundle.exists():
        print(
            f"Error: bundled data.zip not found at {bundle}\n"
            "Run `python -m calvin.app bundle ...` to create it first.",
            file=sys.stderr,
        )
        sys.exit(1)

    tmpdir = Path(tempfile.mkdtemp(prefix="calvin-app-"))
    atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)

    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(tmpdir)

    studies_dir = tmpdir / "studies"
    study_paths: list[Path] = []
    if studies_dir.is_dir():
        study_paths = sorted(
            p for p in studies_dir.iterdir() if p.is_dir()
        )

    return tmpdir, study_paths


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "bundle":
        _cmd_bundle(args)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    # ------------------------------------------------------------------ mode
    if args.hosted:
        if args.data:
            print("Warning: --data is ignored in --hosted mode.", file=sys.stderr)
        if args.studies:
            print("Warning: --study is ignored in --hosted mode.", file=sys.stderr)
        bundle_tmpdir, study_paths = _extract_bundle()
        data_path = None  # not used in prebuilt mode

    elif args.local:
        if args.studies:
            print("Warning: --study is ignored in --local mode (studies are auto-discovered).", file=sys.stderr)
        if not args.data:
            print("Error: --data is required in --local mode.", file=sys.stderr)
            sys.exit(1)
        data_path = Path(args.data)
        base = Path("my-models")
        study_paths = _discover_local_studies(base)
        if not study_paths:
            print(
                f"Warning: no studies found under {base.resolve()}. "
                "The app will run without model results.",
                file=sys.stderr,
            )
    else:
        # Original explicit mode
        if not args.data:
            print("Error: --data is required (or use --local / --hosted).", file=sys.stderr)
            sys.exit(1)
        data_path = Path(args.data)
        study_paths = [Path(s) for s in args.studies]
        if not study_paths:
            default_path = Path("my-models/calvin-pf")
            if default_path.exists():
                study_paths = [default_path]
                print(f"No --study specified; using default: {default_path}")
            else:
                print(
                    "Warning: no --study specified and my-models/calvin-pf not found. "
                    "The app will run without model results.",
                    file=sys.stderr,
                )

    # --------------------------------------------------------- validate paths
    if not args.hosted and not data_path.exists():
        print(f"Error: network data path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------- dependency check
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as e:
        print(
            f"\nError: missing dependency — {e}\n\n"
            "Install the web app dependencies with:\n"
            "  pip install fastapi 'uvicorn[standard]'\n",
            file=sys.stderr,
        )
        sys.exit(1)

    from calvin.app.server import create_app

    if args.hosted:
        app = create_app(
            prebuilt_dir=bundle_tmpdir,
            default_study=args.default_study,
        )
    else:
        app = create_app(
            data_path=data_path,
            study_paths=study_paths,
            default_study=args.default_study,
        )

    # Resolve host/port. In hosted mode always bind to 0.0.0.0 (Render requires it).
    # PORT env var takes priority over --port (Render injects the assigned port).
    if args.hosted:
        host = os.environ.get("HOST", "0.0.0.0")
    else:
        host = os.environ.get("HOST", args.host)
    port = int(os.environ.get("PORT", args.port))

    mode_label = "hosted" if args.hosted else ("local" if args.local else "explicit")
    print(f"\nCalvin Network App starting at http://{host}:{port}  [{mode_label} mode]")
    if args.hosted:
        print(f"  Network data : (pre-built from data.zip)")
    else:
        print(f"  Network data : {data_path}")
    for sp in study_paths:
        print(f"  Study        : {sp}")
    if not study_paths:
        print("  Study        : (none)")
    print("  API docs     : /docs\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )


def _cmd_bundle(args: argparse.Namespace) -> None:
    import json as _json

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Error: --data path does not exist: {data_path}", file=sys.stderr)
        sys.exit(1)

    study_paths = [Path(s).resolve() for s in args.studies]
    for sp in study_paths:
        if not sp.exists():
            print(f"Error: study path does not exist: {sp}", file=sys.stderr)
            sys.exit(1)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).parent / "data.zip"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building {out_path} ...")

    # --- Pre-build network to avoid reading ~7 000 files at hosted startup ---
    print("  Pre-building network (loading from raw data) ...")
    from calvin.network import load_network
    from calvin.app.state import AppState, network_to_dict

    _state = AppState()
    _state.network = load_network(str(data_path))
    _state._build_geojson(data_path)

    geojson_bytes = _json.dumps(_state.geojson).encode()
    network_bytes = _json.dumps(network_to_dict(_state.network)).encode()
    print(
        f"  prebuilt/network.geojson : {len(geojson_bytes) / 1024:.0f} KB  "
        f"({len(_state.geojson['features'])} features)"
    )
    print(
        f"  prebuilt/network.json    : {len(network_bytes) / 1024:.0f} KB  "
        f"({len(_state.network.nodes)} nodes, {len(_state.network.links)} links)"
    )

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write pre-built network files → prebuilt/
        zf.writestr("prebuilt/network.geojson", geojson_bytes)
        zf.writestr("prebuilt/network.json", network_bytes)

        # Write each study's results → studies/{name}/results/ + r-dict.json
        for sp in study_paths:
            results_dir = sp / "results"
            if not results_dir.is_dir():
                print(
                    f"  Warning: {sp} has no results/ subdirectory; skipping.",
                    file=sys.stderr,
                )
                continue
            study_name = sp.name
            csv_count = 0
            for src in results_dir.rglob("*"):
                if src.is_file():
                    arc = f"studies/{study_name}/results/{src.relative_to(results_dir).as_posix()}"
                    zf.write(src, arc)
                    csv_count += 1
            # Also include study-root COSVF files if present
            for fname in ("r-dict.json", "cosvf-params.csv"):
                src = sp / fname
                if src.exists():
                    zf.write(src, f"studies/{study_name}/{fname}")
                    csv_count += 1
            print(f"  studies/{study_name}/results/ : {csv_count} files from {results_dir}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nBundle written: {out_path}  ({size_mb:.1f} MB)")
    print("\nTo serve the bundle:")
    print("  python -m calvin.app serve --hosted")


if __name__ == "__main__":
    main()
