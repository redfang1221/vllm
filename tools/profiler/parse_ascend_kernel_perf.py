# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print selected Ascend profiler kernel performance rows.

The expected input tree is:

    <root>/
      <kernel_name>_perf/
        <run_dir>/
          ASCEND_PROFILER_OUTPUT/
            op_statistic.csv
            kernel_details.csv

For every profiler output directory, this script:
1. Reads snake_case OP Type rows from op_statistic.csv.
2. Finds kernel_details.csv rows with a matching Name.
3. Selects the row whose Duration(us) is closest to Avg Time(us).
4. Prints the requested spec and result fields.
"""

from __future__ import annotations

import argparse
import csv
import re
import os
from pathlib import Path

SNAKE_CASE_RE = re.compile(r"^_*[a-z0-9]+(?:_[a-z0-9]+)*_*$")

SPEC_COLUMNS = [
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
]

RESULT_COLUMNS = [
    "aicore_time(us)",
    "aic_mac_ratio",
    "aic_scalar_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_mte3_ratio",
    "aic_fixpipe_ratio",
    "aiv_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
]

SEPARATOR = "-" * 78
CONTEXT = []

def my_print(string):
    global CONTEXT
    print(string)
    CONTEXT += [string]



def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {
                (key or "").strip(): (value or "").strip()
                for key, value in row.items()
            }
            rows.append(normalized)
        return rows


def _to_float(value: str) -> float | None:
    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _is_snake_case_op(op_type: str) -> bool:
    return bool(SNAKE_CASE_RE.fullmatch(op_type.strip()))


def _find_perf_dir(path: Path, root: Path) -> Path:
    for parent in path.parents:
        if parent == root.parent:
            break
        if parent.name.endswith("_perf"):
            return parent
    return path.parent


def _find_profiler_outputs(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("ASCEND_PROFILER_OUTPUT")
        if (path / "op_statistic.csv").is_file()
        and (path / "kernel_details.csv").is_file()
    )


def _matching_detail_rows(
    detail_rows: list[dict[str, str]],
    op_type: str,
    allow_name_contains: bool,
) -> list[dict[str, str]]:
    exact = [row for row in detail_rows if row.get("Name", "") == op_type]
    if exact or not allow_name_contains:
        return exact
    return [row for row in detail_rows if op_type in row.get("Name", "")]


def _select_closest_duration_row(
    detail_rows: list[dict[str, str]],
    avg_time_us: float,
) -> dict[str, str] | None:
    candidates: list[tuple[float, dict[str, str]]] = []
    for row in detail_rows:
        duration_us = _to_float(row.get("Duration(us)", ""))
        if duration_us is None:
            continue
        candidates.append((abs(duration_us - avg_time_us), row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _print_section(title: str, row: dict[str, str], columns: list[str]) -> None:
    my_print(f"{title}:")
    for column in columns:
        my_print(f"  - {column}: {row.get(column, '')}")


def _print_kernel_result(
    kernel_name: str,
    run_name: str,
    op_type: str,
    avg_time_us: str,
    detail_row: dict[str, str],
    show_source: bool,
) -> None:
    my_print(SEPARATOR)
    my_print(f"kernel: {kernel_name}")
    if show_source:
        my_print(f"run: {run_name}")
        my_print(f"op_type: {op_type}")
        my_print(f"avg_time(us): {avg_time_us}")
        my_print(f"matched_duration(us): {detail_row.get('Duration(us)', '')}")
    _print_section("Specs", detail_row, SPEC_COLUMNS)
    _print_section("Result", detail_row, RESULT_COLUMNS)


def parse_profiler_output(
    profiler_output: Path,
    root: Path,
    allow_name_contains: bool,
    show_source: bool,
) -> int:
    op_stat_rows = _read_csv(profiler_output / "op_statistic.csv")
    detail_rows = _read_csv(profiler_output / "kernel_details.csv")
    perf_dir = _find_perf_dir(profiler_output, root)
    run_name = profiler_output.parent.name
    printed = 0

    for stat_row in op_stat_rows:
        op_type = stat_row.get("OP Type", "")
        if not _is_snake_case_op(op_type):
            continue

        avg_time_text = stat_row.get("Avg Time(us)", "")
        avg_time_us = _to_float(avg_time_text)
        if avg_time_us is None:
            continue

        matches = _matching_detail_rows(detail_rows, op_type,
                                        allow_name_contains)
        selected = _select_closest_duration_row(matches, avg_time_us)
        if selected is None:
            continue

        _print_kernel_result(
            kernel_name=perf_dir.name,
            run_name=run_name,
            op_type=op_type,
            avg_time_us=avg_time_text,
            detail_row=selected,
            show_source=show_source,
        )
        printed += 1

    return printed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Ascend profiler kernel performance CSV outputs.")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory containing one or more <kernel>_perf folders.",
    )
    parser.add_argument(
        "--allow-name-contains",
        action="store_true",
        help=("If no exact kernel_details Name match exists, allow substring "
              "matching against the op_statistic OP Type."),
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Also print run directory, OP Type, Avg Time and matched Duration.",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default="output.csv",
        help="Set output file name.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {root}")

    outputs = _find_profiler_outputs(root)
    if not outputs:
        raise SystemExit(
            "No ASCEND_PROFILER_OUTPUT directories with both "
            "op_statistic.csv and kernel_details.csv were found.")

    total = 0
    for profiler_output in outputs:
        total += parse_profiler_output(
            profiler_output=profiler_output,
            root=root,
            allow_name_contains=args.allow_name_contains,
            show_source=args.show_source,
        )

    if total:
        my_print(SEPARATOR)
        if args.show_source:
            parse_log_to_csv(CONTEXT, args.output_file_name)
    else:
        raise SystemExit("No snake_case OP Type rows matched kernel_details rows.")


def parse_log_to_csv(context_list, output_file):
    file_path = Path(output_file)
    if file_path.exists():
        file_path.unlink()
    headers = [
        "op_type", "avg_time(us)", "Input Shapes", "Input Data Types", "Input Formats",
        "Output Shapes", "Output Data Types", "Output Formats", "aiv_time(us)",
        "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
        "aicore_time(us)", "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio",
        "aic_mte2_ratio", "aic_mte3_ratio", "aic_fixpipe_ratio"
    ]
    data_rows = []
    current_row = {}
    last_key = None
    key_pattern = re.compile(r'^\s*(?:-\s+)?([\w\(\)\s/]+):\s*(.*)')
    for line in context_list:
        line_str = line.strip()
        if re.match(r'-{10,}', line_str):
            if current_row:
                data_rows.append([current_row.get(h, "N/A") for h in headers])
                current_row = {}
                last_key = None
            continue
        if not line_str:
            continue
        match = key_pattern.match(line_str)
        if match:
            k = match.group(1).strip()
            v = match.group(2).strip().strip('"')
            current_row[k] = v
            last_key = k
        else:
            if last_key:
                current_row[last_key] = (current_row[last_key] + " " + line_str).strip()
    if current_row:
        data_rows.append([current_row.get(h, "N/A") for h in headers])
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data_rows)


if __name__ == "__main__":
    main()
