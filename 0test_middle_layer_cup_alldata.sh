pkill -f eval_arx5


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/models/cup_wild_vit_l_1img.ckpt \
#     -o data/experiments/0212_new_datloader_test \
#     --task_name cup \
#     --no_mirror --policy_port 8769 \
    # --match_dataset_path data/experiments/0128-dataall_umi_cup-dataaugtype3-lr1e-5-2-bluecup-16diffstep-eval \
    # --match_episode 0 \
    


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy_new/data/models/new_dataloader/latest.ckpt \
#     -o data/experiments/0212_new_datloader_test \
#     --task_name cup \
#     --no_mirror --policy_port 8769

python scripts/eval_arx5.py \
    -i ../diffusion_policy_new/data/models/new_augmentation/latest.ckpt \
    -o data/experiments/0214_new_augmentation_test \
    --task_name cup \
    --no_mirror --policy_port 8769

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_cup_all_data/dataall_umi_cup-dataaugtype3-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0126-dataall_umi_cup-dataaugtype3-lr1e-5 \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi_cup_all_data/dataall_umi_cup-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
#     -o data/experiments/0131-dataall_umi_cup-dataaugtype3-lr1e-5-2-bluecup-16diffstep-eval \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4 \
#     --task_name cup \
#     --match_dataset_path data/experiments/0128-dataall_umi_cup-dataaugtype3-lr1e-5-2-bluecup-16diffstep-eval \
#     --match_episode 25 \
    

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0126-cup-alldata-prevbest \
#     --different_history_freq \
#     --frequency 5 --steps_per_inference 4 \
#     --task_name cup \
#     --no_mirror --policy_port 8768