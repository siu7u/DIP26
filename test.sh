#!/usr/bin/env bash

DATASETS="ceilnet_table2 real20 postcard objects wild sir2_withgt"

# baseline
# for dataset in $DATASETS
# do
#   python test_errnet.py \
#     --dataset $dataset \
#     --name errnet_base \
#     --hyper \
#     -r \
#     --icnn_path checkpoints/errnet_base/errnet_latest.pt \
#     --result_dir results_ablation \
#     --save_subdir errnet_base_$dataset
# done

# RIP input only
for dataset in $DATASETS
do
  python test_errnet.py \
    --dataset $dataset \
    --name rip_input_only \
    --hyper \
    --use_rpen \
    -r \
    --icnn_path checkpoints/rip_input_only/errnet_latest.pt \
    --result_dir results_ablation \
    --save_subdir rip_input_only_$dataset
done

# RIP prior supervision
for dataset in $DATASETS
do
  python test_errnet.py \
    --dataset $dataset \
    --name rip_prior_sup \
    --hyper \
    --use_rpen \
    -r \
    --icnn_path checkpoints/rip_prior_sup/errnet_latest.pt \
    --result_dir results_ablation \
    --save_subdir rip_prior_sup_$dataset
done

# RIP prior + background
for dataset in $DATASETS
do
  python test_errnet.py \
    --dataset $dataset \
    --name rip_prior_bg \
    --hyper \
    --use_rpen \
    -r \
    --icnn_path checkpoints/rip_prior_bg/errnet_latest.pt \
    --result_dir results_ablation \
    --save_subdir rip_prior_bg_$dataset
done

# RIP full
for dataset in $DATASETS
do
  python test_errnet.py \
    --dataset $dataset \
    --name rip_full \
    --hyper \
    --use_rpen \
    -r \
    --icnn_path checkpoints/rip_full/errnet_latest.pt \
    --result_dir results_ablation \
    --save_subdir rip_full_$dataset
done