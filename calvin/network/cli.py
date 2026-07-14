"""
CLI entry point for calvin network tools.

Replaces the ``cnf`` Node.js CLI. Usage::

    python -m calvin.network.cli matrix --data /path/to/data ...
    python -m calvin.network.cli list --data /path/to/data
    python -m calvin.network.cli validate --data /path/to/data
    python -m calvin.network.cli apply-changes csv --file changes.csv --data /path/to/data

Or if installed as a console script (see pyproject.toml)::

    calvin-network matrix --data /path/to/data ...
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="calvin-network",
        description="CALVIN network data tools — Python replacement for cnf",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- matrix ----
    p_matrix = subparsers.add_parser(
        "matrix",
        help="Build the time-expanded network matrix CSV",
    )
    p_matrix.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_matrix.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout)",
    )
    p_matrix.add_argument(
        "--format", "-f", choices=["csv", "tsv"], default="csv",
        help="Output format (default: csv)",
    )
    p_matrix.add_argument(
        "--start", "-s", default="1921-10",
        help="Start date YYYY-MM (default: 1921-10)",
    )
    p_matrix.add_argument(
        "--stop", "-t", default="2003-09",
        help="Stop date YYYY-MM (default: 2003-09)",
    )
    p_matrix.add_argument(
        "--max-ub", "-M", type=float, default=None,
        help="Replace null upper bounds with this value (default: DEFAULT_MAX_UB from matrix.py)",
    )
    p_matrix.add_argument(
        "--sep", default=".",
        help="Separator between node name and date (default: .)",
    )
    p_matrix.add_argument(
        "--debug", action="store_true",
        help="Add debug source/sink links for infeasibility diagnosis",
    )
    p_matrix.add_argument(
        "--debug-cost", type=float, default=2e7,
        help="Cost for debug links (default: 2e7)",
    )
    p_matrix.add_argument(
        "--regions", "-r", default=None,
        help="Comma-separated list of regions to include",
    )
    p_matrix.add_argument(
        "nodes", nargs="*", default=None,
        help="Specific nodes to include (omit for full network)",
    )

    # ---- list ----
    p_list = subparsers.add_parser(
        "list",
        help="List nodes and/or links in the data repository",
    )
    p_list.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_list.add_argument(
        "--type", choices=["all", "nodes", "links"], default="all",
        help="What to list (default: all)",
    )
    p_list.add_argument(
        "names", nargs="*", default=None,
        help="Filter by prmname (omit for all)",
    )

    # ---- validate ----
    p_validate = subparsers.add_parser(
        "validate",
        help="Validate the network data repository",
    )
    p_validate.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_validate.add_argument(
        "--dump-csv", default=None,
        help="Write errors to CSV file",
    )

    # ---- apply-changes ----
    p_apply = subparsers.add_parser(
        "apply-changes",
        help="Apply external data changes to the repository",
    )
    p_apply_sub = p_apply.add_subparsers(dest="apply_type", required=True)

    # apply-changes csv
    p_apply_csv = p_apply_sub.add_parser(
        "csv",
        help="Import timeseries from a CSV file",
    )
    p_apply_csv.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_apply_csv.add_argument(
        "--file", "-f", required=True,
        help="Path to CSV file with columns: date, PRMNAME1, PRMNAME2, ...",
    )
    p_apply_csv.add_argument(
        "--property", "-p", choices=["flow", "storage", "inflows"],
        default="flow",
        help="Which property to update (default: flow)",
    )
    p_apply_csv.add_argument(
        "--dry-run", action="store_true",
        help="Only report changes, don't write files",
    )

    # apply-changes excel
    p_apply_excel = p_apply_sub.add_parser(
        "excel",
        help="Import data from an Excel workbook",
    )
    p_apply_excel.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_apply_excel.add_argument(
        "--file", "-f", required=True,
        help="Path to .xlsx file",
    )
    p_apply_excel.add_argument(
        "--dry-run", action="store_true",
        help="Only report changes, don't write files",
    )

    # ---- regions ----
    p_regions = subparsers.add_parser(
        "regions",
        help="List regions in the data repository",
    )
    p_regions.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )

    # ---- prepare-cosvf ----
    p_cosvf = subparsers.add_parser(
        "prepare-cosvf",
        help="Prepare input files for a COSVF limited-foresight run",
    )
    p_cosvf.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_cosvf.add_argument(
        "--output", "-o", required=True,
        help="Output directory for COSVF model files (e.g. ./my-models/calvin-cosvf)",
    )
    p_cosvf.add_argument(
        "--start", "-s", default="1921-10",
        help="Full period start YYYY-MM (default: 1921-10)",
    )
    p_cosvf.add_argument(
        "--stop", "-t", default="2003-09",
        help="Full period stop YYYY-MM (default: 2003-09)",
    )
    p_cosvf.add_argument(
        "--wy1-start", default="1921-10",
        help="First water year start YYYY-MM (default: 1921-10)",
    )
    p_cosvf.add_argument(
        "--wy1-stop", default="1922-09",
        help="First water year stop YYYY-MM (default: 1922-09)",
    )

    # ---- prepare-pf-astep ----
    p_pf_astep = subparsers.add_parser(
        "prepare-pf-astep",
        help="Prepare annual-step perfect foresight links file (82 annual timesteps)",
    )
    p_pf_astep.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_pf_astep.add_argument(
        "--output", "-o", required=True,
        help="Output directory for the links.csv file",
    )
    p_pf_astep.add_argument(
        "--start", "-s", default="1921-10",
        help="Full period start YYYY-MM (default: 1921-10)",
    )
    p_pf_astep.add_argument(
        "--stop", "-t", default="2003-09",
        help="Full period stop YYYY-MM (default: 2003-09)",
    )

    # ---- prepare-cosvf-astep ----
    p_cosvf_astep = subparsers.add_parser(
        "prepare-cosvf-astep",
        help="Prepare annual-step COSVF input files (1-step template + supporting files)",
    )
    p_cosvf_astep.add_argument(
        "--data", "-d", required=True,
        help="Path to calvin-network-data/data directory",
    )
    p_cosvf_astep.add_argument(
        "--output", "-o", required=True,
        help="Output directory for COSVF model files",
    )
    p_cosvf_astep.add_argument(
        "--start", "-s", default="1921-10",
        help="Full period start YYYY-MM (default: 1921-10)",
    )
    p_cosvf_astep.add_argument(
        "--stop", "-t", default="2003-09",
        help="Full period stop YYYY-MM (default: 2003-09)",
    )
    p_cosvf_astep.add_argument(
        "--wy1-start", default="1921-10",
        help="First (template) water year start YYYY-MM (default: 1921-10)",
    )
    p_cosvf_astep.add_argument(
        "--wy1-stop", default="1922-09",
        help="First (template) water year stop YYYY-MM (default: 1922-09)",
    )

    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(message)s",
    )

    # Dispatch commands
    if args.command == "matrix":
        _cmd_matrix(args)
    elif args.command == "list":
        _cmd_list(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "apply-changes":
        _cmd_apply_changes(args)
    elif args.command == "regions":
        _cmd_regions(args)
    elif args.command == "prepare-cosvf":
        _cmd_prepare_cosvf(args)
    elif args.command == "prepare-pf-astep":
        _cmd_prepare_pf_astep(args)
    elif args.command == "prepare-cosvf-astep":
        _cmd_prepare_cosvf_astep(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_matrix(args):
    from .loader import load_network
    from .matrix import DEFAULT_MAX_UB, build_matrix, export_matrix
    from .prepare import NODE_LB_OVERRIDES

    nodes = args.nodes if args.nodes else None
    regions = args.regions.split(",") if args.regions else None

    network = load_network(args.data, nodes=nodes, regions=regions)
    df = build_matrix(
        network,
        start=args.start,
        stop=args.stop,
        max_ub=args.max_ub if args.max_ub is not None else DEFAULT_MAX_UB,
        sep=args.sep,
        add_debug=args.debug,
        debug_cost=args.debug_cost,
        node_lb_overrides=NODE_LB_OVERRIDES,
    )

    if args.output:
        export_matrix(df, output=args.output, fmt=args.format)
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        result = export_matrix(df, fmt=args.format)
        sys.stdout.write(result)


def _cmd_list(args):
    from .query import list_items

    names = args.names if args.names else None
    df = list_items(args.data, item_type=args.type, filter_names=names)
    if df.empty:
        print("No items found.")
    else:
        print(df.to_string(index=False))


def _cmd_validate(args):
    import csv as csv_mod

    from .validate import validate_data

    result = validate_data(args.data)
    print(result.summary())
    for err in result.errors:
        print(f"  {err}")

    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as f:
            writer = csv_mod.writer(f)
            writer.writerow(["severity", "path", "message"])
            for err in result.errors:
                writer.writerow([err.severity, err.path, err.message])
        print(f"Errors written to {args.dump_csv}")

    sys.exit(0 if result.is_valid else 1)


def _cmd_apply_changes(args):
    if args.apply_type == "csv":
        from .apply_changes import apply_csv
        files = apply_csv(
            args.file, args.data, prop=args.property, dry_run=args.dry_run
        )
        print(f"{'Would update' if args.dry_run else 'Updated'} {len(files)} files")

    elif args.apply_type == "excel":
        from .apply_changes import apply_excel
        files = apply_excel(args.file, args.data, dry_run=args.dry_run)
        print(f"{'Would update' if args.dry_run else 'Updated'} {len(files)} files")


def _cmd_regions(args):
    from .query import list_regions

    regions = list_regions(args.data)
    if not regions:
        print("No regions found.")
    else:
        for r in regions:
            print(f"{r['name']:30s} {r['path']}")


def _cmd_prepare_cosvf(args):
    from .prepare import prepare_cosvf

    out = prepare_cosvf(
        data_path=args.data,
        output_dir=args.output,
        start=args.start,
        stop=args.stop,
        wy1_start=args.wy1_start,
        wy1_stop=args.wy1_stop,
    )
    print(f"COSVF input files written to {out}")


def _cmd_prepare_pf_astep(args):
    from .prepare import prepare_pf_astep

    out = prepare_pf_astep(
        data_path=args.data,
        output_dir=args.output,
        start=args.start,
        stop=args.stop,
    )
    print(f"Annual-step PF links written to {out}")


def _cmd_prepare_cosvf_astep(args):
    from .prepare import prepare_cosvf_astep

    out = prepare_cosvf_astep(
        data_path=args.data,
        output_dir=args.output,
        start=args.start,
        stop=args.stop,
        wy1_start=args.wy1_start,
        wy1_stop=args.wy1_stop,
    )
    print(f"Annual-step COSVF input files written to {out}")


if __name__ == "__main__":
    main()
