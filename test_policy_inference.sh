python scripts/run_policy_inference.py \
    --ckpt_path ../diffusion_policy_new/data/outputs_v5/envrobodiff-noval-data500_umi_multi-lr1e-5-marlowe-bs32-2mode-refactor/checkpoints/latest.ckpt \
    --policy_ip 127.0.0.1 --policy_port 8768 \
    --data_dir data/experiments/0124-data500_umi_mix_dataaug-lr1e-5-2
# python scripts/run_policy_inference.py \
#     --ckpt_path ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/0diffusion_policy_data500_umi_mix_-dataaug3-noactnorm-lang/checkpoints/latest.ckpt \
#     --policy_ip 127.0.0.1 --policy_port 8769 \
#     --data_dir data/experiments/0127-0diffusion_policy_data500_umi_mix_-dataaug3-noactnorm-lang-eval