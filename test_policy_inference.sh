python scripts/run_policy_inference.py \
    --ckpt_path ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-2/checkpoints/latest.ckpt \
    --policy_ip 127.0.0.1 --policy_port 8768 \
    --data_dir data/experiments/0124-data500_umi_mix_dataaug-lr1e-5-2