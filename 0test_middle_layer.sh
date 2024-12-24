# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/models/cup_wild_vit_l_1img.ckpt \
#     -o data/experiments/1123 \
#     --no_mirror --policy_port 8769;

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-historyact-mask0.9-2/checkpoints/latest.ckpt \
#     -o data/experiments/1123 \
#     --no_mirror --policy_port 8768;

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-proprioception-2/checkpoints/latest.ckpt \
#     -o data/experiments/1220 \
#     --no_mirror --policy_port 8768 --policy_ip 172.24.95.38

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj/checkpoints/latest.ckpt \
#     -o data/experiments/1219 \
#     --no_mirror --policy_port 8768;

python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-proprioception-newact/checkpoints/latest.ckpt \
    -o data/experiments/1223 \
    --no_mirror --policy_port 8768 
    # --policy_ip 172.24.95.38

