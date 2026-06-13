import argparse
import json
import os
import sys
from os.path import join

import numpy as np
import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
from engine import Engine
from options.errnet.train_options import TrainOptions


EVAL_DATASETS = {
    "ceilnet_table2": {
        "dataset_name": "testdata_table2",
        "path": "testdata_CEILNET_table2",
        "save_subdir": "CEILNet_table2",
    },
    "real20": {
        "dataset_name": "testdata_real",
        "path": "real20",
        "save_subdir": "real20",
        "max_long_edge": 512,
    },
    "postcard": {
        "dataset_name": "testdata_postcard",
        "path": "postcard",
        "save_subdir": "postcard",
    },
    "objects": {
        "dataset_name": "testdata_objects",
        "path": "objects",
        "save_subdir": "objects",
    },
    "wild": {
        "dataset_name": "testdata_wild",
        "path": "wild",
        "save_subdir": "wild",
    },
    "sir2_withgt": {
        "dataset_name": "testdata_sir2",
        "path": "sir2_withgt",
        "save_subdir": "sir2_withgt",
    },
}

TEST_DATASETS = {
    "internet": {
        "path": None,
        "save_subdir": "internet",
    },
}


def parse_test_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(list(EVAL_DATASETS.keys()) + list(TEST_DATASETS.keys()) + ["custom"]),
        help="dataset to run",
    )
    parser.add_argument(
        "--data_root",
        default="./datasets/processed_data",
        help="root directory for processed datasets",
    )
    parser.add_argument(
        "--input_dir",
        default="./datasets/raw_data/CEILNet/testdata_reflection_real",
        help="input directory for test-only datasets such as internet/custom images",
    )
    parser.add_argument(
        "--result_dir",
        default="./results",
        help="directory for saved outputs",
    )
    parser.add_argument(
        "--save_subdir",
        default=None,
        help="override output subdirectory name under result_dir",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=None,
        help="resize test images so the longest edge does not exceed this value",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def build_eval_dataloader(opt, data_root, dataset_key, max_long_edge=None):
    spec = EVAL_DATASETS[dataset_key]
    dataset = datasets.CEILTestDataset(
        join(data_root, spec["path"]),
        max_long_edge=max_long_edge if max_long_edge is not None else spec.get("max_long_edge"),
    )
    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True,
    )
    return spec, dataloader


def build_test_dataloader(opt, dataset_key, input_dir, max_long_edge=None):
    if dataset_key == "custom":
        dataset = datasets.RealDataset(input_dir, max_long_edge=max_long_edge)
        save_subdir = "custom"
    else:
        spec = TEST_DATASETS[dataset_key]
        dataset = datasets.RealDataset(
            input_dir if spec["path"] is None else join(input_dir, spec["path"]),
            max_long_edge=max_long_edge,
        )
        save_subdir = spec["save_subdir"]

    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True,
    )
    return save_subdir, dataloader


def meters_to_dict(avg_meters):
    return {key: to_jsonable(avg_meters[key]) for key in avg_meters.keys()}


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def update_summary_json(result_dir, save_subdir, metrics=None, run_info=None):
    os.makedirs(result_dir, exist_ok=True)
    summary_path = join(result_dir, "metrics.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}

    entry = {}
    if metrics is not None:
        entry["metrics"] = to_jsonable(metrics)
    if run_info is not None:
        entry["run_info"] = to_jsonable(run_info)
    summary[save_subdir] = entry

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def save_metrics(savedir, result_dir, save_subdir, metrics, run_info=None):
    os.makedirs(savedir, exist_ok=True)
    update_summary_json(result_dir, save_subdir, metrics=metrics, run_info=run_info)

    with open(join(savedir, "metrics.txt"), "w") as f:
        if run_info is not None:
            f.write("[run_info]\n")
            for key in sorted(run_info.keys()):
                f.write("{}: {}\n".format(key, run_info[key]))
            f.write("\n[metrics]\n")
        for key in sorted(metrics.keys()):
            f.write("{}: {:.6f}\n".format(key, metrics[key]))


def save_run_info(savedir, result_dir, save_subdir, run_info):
    os.makedirs(savedir, exist_ok=True)
    update_summary_json(result_dir, save_subdir, run_info=run_info)
    with open(join(savedir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, sort_keys=True)


def main():
    cli_args = parse_test_args()
    option_parser = TrainOptions()
    option_parser.isTrain = False
    opt = option_parser.parse()
    opt.isTrain = False

    cudnn.benchmark = len(opt.gpu_ids) > 0
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False

    engine = Engine(opt)

    if cli_args.dataset in EVAL_DATASETS:
        spec, dataloader = build_eval_dataloader(
            opt,
            cli_args.data_root,
            cli_args.dataset,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or spec["save_subdir"]
        savedir = join(cli_args.result_dir, save_subdir)
        res = engine.eval(
            dataloader,
            dataset_name=spec["dataset_name"],
            savedir=savedir,
        )
        metrics = meters_to_dict(res)
        save_metrics(
            savedir,
            cli_args.result_dir,
            save_subdir,
            metrics,
            run_info={
                "dataset": cli_args.dataset,
                "data_root": cli_args.data_root,
                "model_name": opt.name,
                "checkpoint": opt.icnn_path,
                "use_rpen": getattr(opt, "use_rpen", False),
                "hyper": getattr(opt, "hyper", False),
                "max_long_edge": cli_args.max_long_edge,
            },
        )
        print(res)
    else:
        default_save_subdir, dataloader = build_test_dataloader(
            opt,
            cli_args.dataset,
            cli_args.input_dir,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or default_save_subdir
        savedir = join(cli_args.result_dir, save_subdir)
        engine.test(
            dataloader,
            savedir=savedir,
        )
        save_run_info(
            savedir,
            cli_args.result_dir,
            save_subdir,
            {
                "dataset": cli_args.dataset,
                "input_dir": cli_args.input_dir,
                "model_name": opt.name,
                "checkpoint": opt.icnn_path,
                "use_rpen": getattr(opt, "use_rpen", False),
                "hyper": getattr(opt, "hyper", False),
                "max_long_edge": cli_args.max_long_edge,
                "note": "No metrics are available for test-only datasets without ground truth.",
            },
        )


if __name__ == "__main__":
    main()
