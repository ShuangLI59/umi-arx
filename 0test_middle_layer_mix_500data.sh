
# sudo chmod 777 -R /dev/bus/usb/
pkill -f eval_arx5

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
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2-mouse \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-5/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-5-mouse \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name mouse

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-5-then-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-5-then-lr1e-5 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype4-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype4-lr1e-5-mouse \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup



# =========================================================================
#                               Jan 26
# =========================================================================

# ====================Mixed====================
# DP
python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/0diffusion_policy_data500_umi_mix_-dataaug3-noactnorm/checkpoints/latest.ckpt \
    -o data/experiments/0127-0diffusion_policy_data500_umi_mix_-dataaug3-noactnorm-80epoch-cup-eval \
    --policy_port 8769 \
    --frequency 5 --steps_per_inference 4 \
    --task_name cup \
    --match_dataset_path data/experiments/0126-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5-cup-eval \
    --match_episode 0

    # --match_dataset_path data/experiments/0126-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5-towel-eval-final \

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/0diffusion_policy_data500_umi_mix_-dataaug3-noactnorm-lang/checkpoints/latest.ckpt \
#     -o data/experiments/0127-0diffusion_policy_data500_umi_mix_-dataaug3-noactnorm-lang-eval \
#     --policy_port 8769 \
#     --frequency 5 --steps_per_inference 4 \
#     --match_dataset_path data/experiments/0126-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5-cup-eval-final \
#     --match_episode 0 \
#     --task_name cup


# Ours
#### For evaluation
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5/checkpoints/latest.ckpt \
#     -o data/experiments/0126-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5-towel-eval-black-gripper \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name towel

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-5-then-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0127-data500_umi_mix_dataaug-dataaugtype3-5-then-lr1e-5-cup-eval \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype4-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0127-round2-data500_umi_mix_dataaug-dataaugtype4-lr1e-5-cup-eval-orangegripper \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
#     -o data/experiments/0126-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2-cup-eval \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup
