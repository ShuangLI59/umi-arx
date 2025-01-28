pkill -f eval_arx5


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/models/cup_wild_vit_l_1img.ckpt \
#     -o data/experiments/0119_umi_cup_official \
#     --no_mirror --policy_port 8769;

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_cup_all_data/dataall_umi_cup-dataaugtype3-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0126-dataall_umi_cup-dataaugtype3-lr1e-5 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_cup_all_data/dataall_umi_cup-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
#     -o data/experiments/0126-dataall_umi_cup-dataaugtype3-lr1e-5-2 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0126-cup-alldata-prevbest \
#     --different_history_freq \
#     --frequency 5 --steps_per_inference 4 \
#     --task_name cup \
#     --no_mirror --policy_port 8768