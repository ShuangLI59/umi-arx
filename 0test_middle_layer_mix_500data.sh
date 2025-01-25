
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
python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
    -o data/experiments/0124-data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2-mouse \
    --different_history_freq \
    --policy_port 8768 --frequency 5 --steps_per_inference 4 \
    --task_name towel

# --task_name towel
# --task_name mouse