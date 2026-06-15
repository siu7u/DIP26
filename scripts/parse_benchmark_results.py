#!/usr/bin/env python3
"""Parse logs/benchmark_results.txt into a summary table."""
import re
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "benchmark_results.txt"
OUT = Path(__file__).resolve().parents[1] / "logs" / "benchmark_summary.txt"

METRIC_RE = re.compile(
    r"LMSE:\s*([\d.]+)\s*\|\s*NCC:\s*([\d.]+)\s*\|\s*PSNR:\s*([\d.]+)\s*\|\s*SSIM:\s*([\d.]+)"
)
SECTION_RE = re.compile(r"^===== (.+?) \| (.+?) =====")


def parse(path: Path):
    results = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            current = (m.group(1), m.group(2))
            continue
        m = METRIC_RE.search(line)
        if m and current:
            lmse, ncc, psnr, ssim = m.groups()
            results[current] = {
                "LMSE": float(lmse),
                "NCC": float(ncc),
                "PSNR": float(psnr),
                "SSIM": float(ssim),
            }
    return results


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG
    if not log.exists():
        print(f"Missing {log}")
        sys.exit(1)
    results = parse(log)
    labels = sorted({k[0] for k in results})
    datasets = sorted({k[1] for k in results})

    lines = [f"Benchmark summary ({len(results)} runs)", ""]
    header = f"{'Model':<22}" + "".join(f"{d:>14}" for d in datasets)
    lines.append(header)
    lines.append("-" * len(header))

    for label in labels:
        row = f"{label:<22}"
        for ds in datasets:
            r = results.get((label, ds))
            row += f"{r['PSNR']:>14.2f}" if r else f"{'—':>14}"
        lines.append(row)

    lines.extend(["", "Full metrics (PSNR / SSIM / LMSE / NCC):", ""])
    for label in labels:
        lines.append(f"## {label}")
        for ds in datasets:
            r = results.get((label, ds))
            if r:
                lines.append(
                    f"  {ds:16} PSNR={r['PSNR']:.4f} SSIM={r['SSIM']:.4f} "
                    f"LMSE={r['LMSE']:.4f} NCC={r['NCC']:.4f}"
                )
        lines.append("")

    text = "\n".join(lines)
    OUT.write_text(text)
    print(text)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
