#!/usr/bin/env python3
"""Generate side-by-side comparison figures for report (input / GT / baseline / RDNet / masks)."""
import argparse
import os
import sys
from os.path import basename, join, splitext
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.reflect_dataset as datasets
import models
from options.errnet.train_options import TrainOptions
from util.util import tensor2im

DATASETS = {
    "ceilnet_table2": {
        "path": "testdata_CEILNET_table2",
        "samples": ["2007_003506.png", "2007_004049.png", "2007_006587.png"],
        "rdnet_name": "errnet_rdnet_no_sup",
        "rdnet_ckpt": "checkpoints/errnet_rdnet_no_sup/errnet_model_latest.pt",
        "rdnet_label": "aligned_no_sup",
    },
    "real20": {
        "path": "real20",
        "max_long_edge": 512,
        "samples": ["103.jpg", "12.jpg", "110.jpg"],
        "rdnet_name": "errnet_rdnet_full_unaligned",
        "rdnet_ckpt": "checkpoints/errnet_rdnet_full_unaligned/errnet_model_latest.pt",
        "rdnet_label": "unaligned_full",
    },
}

BASELINE = {
    "name": "errnet",
    "ckpt": "checkpoints/errnet/errnet_060_00463920.pt",
    "label": "baseline",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default="./datasets/processed_data")
    p.add_argument("--result_dir", default="./results/rdnet_comparison")
    p.add_argument("--gpu_ids", default="0")
    return p.parse_known_args()


def build_opt(name, ckpt, use_rdnet, gpu_ids):
    argv = [
        "visualize_rdnet_comparison.py",
        "--name",
        name,
        "--gpu_ids",
        gpu_ids,
        "--hyper",
        "-r",
        "--icnn_path",
        ckpt,
        "--no-verbose",
        "--no-log",
    ]
    if use_rdnet:
        argv.append("--use_rdnet")
    sys.argv = argv
    opt = TrainOptions().parse()
    opt.isTrain = False
    return opt


def load_model(name, ckpt, use_rdnet, gpu_ids):
    opt = build_opt(name, ckpt, use_rdnet, gpu_ids)
    model = models.__dict__[opt.model]()
    model.initialize(opt)
    model._eval()
    return model


@torch.no_grad()
def infer(model, data):
    model.set_input(data, "eval")
    model.forward()
    out = {
        "input": tensor2im(model.input).astype(np.uint8),
        "output": tensor2im(model.output_i).astype(np.uint8),
        "target": tensor2im(model.target_t).astype(np.uint8),
    }
    if getattr(model, "reflection_mask", None) is not None:
        out["pred_mask"] = tensor2im(model.reflection_mask).astype(np.uint8)
    if getattr(model, "target_reflection_mask", None) is not None:
        out["pseudo_mask"] = tensor2im(model.target_reflection_mask).astype(np.uint8)
    return out


def resize_to_height(img, height):
    w, h = img.size
    if h == height:
        return img
    new_w = max(1, int(w * height / h))
    return img.resize((new_w, height), Image.BILINEAR)


def add_caption(img, text, bar_h=28):
    out = Image.new("RGB", (img.size[0], img.size[1] + bar_h), (30, 30, 30))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), text, fill=(240, 240, 240), font=font)
    return out


def make_row(panels):
    height = max(p.size[1] for p in panels)
    resized = [resize_to_height(p, height) for p in panels]
    gap = 4
    total_w = sum(p.size[0] for p in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, height), (20, 20, 20))
    x = 0
    for p in resized:
        canvas.paste(p, (x, 0))
        x += p.size[0] + gap
    return canvas


def to_pil(arr):
    return Image.fromarray(arr)


def visualize_sample(baseline_model, rdnet_model, data, out_path, rdnet_label):
    b = infer(baseline_model, data)
    r = infer(rdnet_model, data)

    panels = [
        add_caption(to_pil(b["input"]), "Input"),
        add_caption(to_pil(b["target"]), "GT"),
        add_caption(to_pil(b["output"]), "Baseline"),
        add_caption(to_pil(r["output"]), f"RDNet ({rdnet_label})"),
    ]
    if "pred_mask" in r:
        panels.append(add_caption(to_pil(r["pred_mask"]), "Pred mask"))
    if "pseudo_mask" in r:
        panels.append(add_caption(to_pil(r["pseudo_mask"]), "MaxRF pseudo"))

    row = make_row(panels)
    row.save(out_path)
    return out_path


def main():
    cli, _ = parse_args()
    cudnn.benchmark = cli.gpu_ids != "-1"

    baseline = load_model(BASELINE["name"], BASELINE["ckpt"], False, cli.gpu_ids)

    for ds_key, spec in DATASETS.items():
        rdnet = load_model(spec["rdnet_name"], spec["rdnet_ckpt"], True, cli.gpu_ids)
        datadir = join(cli.data_root, spec["path"])
        dataset = datasets.CEILTestDataset(
            datadir,
            fns=spec["samples"],
            max_long_edge=spec.get("max_long_edge"),
        )
        loader = datasets.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        ds_out = join(cli.result_dir, ds_key)
        os.makedirs(ds_out, exist_ok=True)

        saved = []
        for data in loader:
            fn = data["fn"][0]
            stem = splitext(basename(fn))[0]
            out_path = join(ds_out, f"{stem}_compare.png")
            visualize_sample(
                baseline,
                rdnet,
                data,
                out_path,
                spec["rdnet_label"],
            )
            saved.append(out_path)
            print(f"saved {out_path}")

        # stack sample rows vertically for one-page figure
        if saved:
            rows = [Image.open(p) for p in saved]
            w = max(r.size[0] for r in rows)
            h = sum(r.size[1] for r in rows) + 4 * (len(rows) - 1)
            sheet = Image.new("RGB", (w, h), (10, 10, 10))
            y = 0
            for r in rows:
                sheet.paste(r, (0, y))
                y += r.size[1] + 4
            sheet_path = join(ds_out, f"{ds_key}_summary.png")
            sheet.save(sheet_path)
            print(f"saved {sheet_path}")

        del rdnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
