pkill -f eval_arx5.py

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_200/0diffusion_policy_data200_umi_cup_1000epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0114 \
#     --no_mirror --policy_port 8769;

python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi_200/data200_umi_cup_1000epoch/checkpoints/latest.ckpt \
    -o data/experiments/0119_ours \
    --different_history_freq \
    --no_mirror --policy_port 8768 

