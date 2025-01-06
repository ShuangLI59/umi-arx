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

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-proprioception-newact/checkpoints/latest.ckpt \
#     -o data/experiments/1223 \
#     --no_mirror --policy_port 8768 
#     # --policy_ip 172.24.95.38

    # -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-proprioception-newact-newdl/checkpoints/epoch=0000-val_loss=0.083.ckpt \

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-proprioception-newact-newdl/checkpoints/latest.ckpt \
#     -o data/experiments/1224_new \
#     --no_mirror --policy_port 8768 


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-fc-newdl-noshiftact-mar_tiny-6actlayer-6imagelayer-repcur/checkpoints/latest.ckpt \
#     -o data/experiments/1228_new \
#     --no_mirror --policy_port 8768 

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-lr1e-5-2/checkpoints/latest.ckpt \
#     -o data/experiments/1230 \
#     --no_mirror --policy_port 8768 


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone/checkpoints/latest.ckpt \
#     -o data/experiments/1228_new \
#     --no_mirror --policy_port 8768 



####################################################################################################################
####################################################################################################################
####################################################################################################################

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-1obs2prop-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/1231 \
#     --short_history \
#     --no_mirror --policy_port 8768 
    

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-lr1e-5/checkpoints/epoch=0007-val_loss=0.067.ckpt \
#     -o data/experiments/1231 \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-1obs2prop-no_mirror-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/1231 \
#     --short_history \
#     --no_mirror --policy_port 8768 
    


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/1231 \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 
    

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-lr1e-5-lrdecay/checkpoints/latest.ckpt \
#     -o data/experiments/1231 \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 
    

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0103 \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 
    

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0103 \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 
    

python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-type2-lr1e-5/checkpoints/latest.ckpt \
    -o data/experiments/0103 \
    --different_history_freq \
    --no_mirror --policy_port 8768 
    