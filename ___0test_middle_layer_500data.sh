
# sudo chmod 777 -R /dev/bus/usb/
pkill -f eval_arx5

# DP official
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/models/cup_wild_vit_l_1img.ckpt \
#     -o data/experiments/0119_umi_cup_official \
#     --no_mirror --policy_port 8769 \
#     --frequency 5 --steps_per_inference 4;


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
#     --frequency 5 --steps_per_inference 8\
#     --no_mirror

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/0diffusion_policy_data500_umi_mix_epoch500/checkpoints/latest.ckpt \
#     -o data/experiments/0220_500data \
#     --policy_port 8769 \
#     --frequency 5 --steps_per_inference 8 \
#     --no_mirror

# ## ours
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --no_mirror \

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix_100epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --no_mirror

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix_200epoch-3/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --no_mirror \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix_100epoch_hisinterFalse/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 
#     # --no_mirror \

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/data500_umi_mix_then_50epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     # --no_mirror


# # ------------------------- mix language -------------------------------------------------------
## DP
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/0diffusion_policy_data500_umi_mix_lang100epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0124_500data_dp \
#     --policy_port 8769 \
#     --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata/0diffusion_policy_data500_umi_mix_lang300epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0124_500data_dp_300epoch \
#     --policy_port 8769 \
#     --frequency 5 --steps_per_inference 4 \
#     --task_name cup


## ours
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype2/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_200episodes_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-lr1e-5-2/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-lr1e-5-2 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
    -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2-mouse \
    --different_history_freq \
    --policy_port 8768 --frequency 5 --steps_per_inference 4 \
    --task_name mouse


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-4/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-4 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


# --task_name towel
# --task_name mouse