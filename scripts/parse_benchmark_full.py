#!/usr/bin/env python3
"""Merge benchmark logs and print full comparison tables."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = [
    ROOT / "logs" / "benchmark_results.txt",
    ROOT / "logs" / "benchmark_unaligned.txt",
]
OUT = ROOT / "logs" / "benchmark_full_summary.txt"

METRIC_RE = re.compile(
    r"LMSE:\s*([\d.]+)\s*\|\s*NCC:\s*([\d.]+)\s*\|\s*PSNR:\s*([\d.]+)\s*\|\s*SSIM:\s*([\d.]+)"
)
SECTION_RE = re.compile(r"^===== (.+?) \| (.+?) =====")


def parse_files(paths):
    results = {}
    for path in paths:
        if not path.exists():
            continue
        current = None
        for line in path.read_text(errors="replace").splitlines():
            m = SECTION_RE.match(line.strip())
            if m:
                label, ds = m.group(1), m.group(2)
                if label.startswith("DONE"):
                    continue
                current = (label, ds)
                continue
            m = METRIC_RE.search(line)
            if m and current:
                lmse, ncc, psnr, ssim = map(float, m.groups())
                results[current] = dict(LMSE=lmse, NCC=ncc, PSNR=psnr, SSIM=ssim)
    return results


def table(results, metric, labels, datasets):
    lines = [f"## {metric} (↑ PSNR/SSIM/NCC, ↓ LMSE)", ""]
    header = f"{'Model':<22}" + "".join(f"{d:>12}" for d in datasets)
    lines += [header, "-" * len(header)]
    for label in labels:
        row = f"{label:<22}"
        for ds in datasets:
            r = results.get((label, ds))
            if r:
                v = r[metric]
                row += f"{v:>12.4f}" if metric == "LMSE" else f"{v:>12.2f}"
            else:
                row += f"{'—':>12}"
        lines.append(row)
    lines.append("")
    return lines


def main():
    results = parse_files(LOGS)
    label_order = [
        "baseline",
        "aligned_no_sup",
        "aligned_sup",
        "aligned_full",
        "unaligned_no_sup",
        "unaligned_sup",
        "unaligned_full",
    ]
    datasets = ["ceilnet_table2", "real20", "objects", "postcard", "wild"]
    labels = [l for l in label_order if any((l, d) in results for d in datasets)]

    lines = [f"Full benchmark summary ({len(results)} runs)", ""]
    for metric in ["PSNR", "SSIM", "LMSE", "NCC"]:
        lines.extend(table(results, metric, labels, datasets))

    # vs baseline delta (ceilnet + real20)
    for metric in ["PSNR", "SSIM", "LMSE", "NCC"]:
        lines.append(f"## {metric} vs baseline (Δ)")
        lines.append("")
        for ds in ["ceilnet_table2", "real20"]:
            base = results.get(("baseline", ds), {}).get(metric)
            if base is None:
                continue
            fmt = f"baseline {metric}={base:.4f}" if metric == "LMSE" else f"baseline {metric}={base:.4f}"
            lines.append(f"### {ds} ({fmt})")
            for label in labels:
                if label == "baseline":
                    continue
                v = results.get((label, ds), {}).get(metric)
                if v is not None:
                    delta = v - base
                    lines.append(f"  {label:<22} {v:8.4f}  ({delta:+.4f})")
            lines.append("")

    text = "\n".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
