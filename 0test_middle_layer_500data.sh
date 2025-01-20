
# sudo chmod 777 -R /dev/bus/usb/
pkill -f eval_arx5

## DP official
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/models/cup_wild_vit_l_1img.ckpt \
#     -o data/experiments/0119_umi_cup_official \
#     --no_mirror --policy_port 8769;


# # ------------------------- cup -------------------------------------------------------
# ## DP
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/0diffusion_policy_data500_umi_cup_epoch500/checkpoints/latest.ckpt \
#     -o data/experiments/0220_500data \
#     --no_mirror \
#     --policy_port 8769;


# ## ours
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_cup/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --no_mirror \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4

# # ------------------------- mix -------------------------------------------------------
# # ## DP
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/0diffusion_policy_data500_umi_mix_epoch100/checkpoints/latest.ckpt \
#     -o data/experiments/0220_500data \
#     --policy_port 8769 \
#     # --no_mirror

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/0diffusion_policy_data500_umi_mix_epoch500/checkpoints/latest.ckpt \
#     -o data/experiments/0220_500data \
#     --policy_port 8769 \
#     # --no_mirror

# ## ours
python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix/checkpoints/latest.ckpt \
    -o data/experiments/0118_mix_200episodes_ours \
    --different_history_freq \
    --policy_port 8768 --frequency 5 --steps_per_inference 4 \
    # --no_mirror \

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix_100epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     # --no_mirror

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix_200epoch-3/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --no_mirror \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4