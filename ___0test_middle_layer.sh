# sudo chmod 777 -R /dev/bus/usb/
pkill -f eval_arx5
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/models/cup_wild_vit_l_1img.ckpt \
#     -o data/experiments/0119_umi_cup_official \
#     --no_mirror --policy_port 8769;

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-historyact-mask0.9-2/checkpoints/latest.ckpt \
#     -o data/experiments/1123 \
#     --no_mirror --policy_port 8768;

# python scripts/eval_arx5.py \
#     -i /home/shuang/shuang/Data-Scaling-Laws/data/models/fold_towel.ckpt \
#     -o data/experiments/0110_mouse \
#     --policy_port 8769;


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
#     -o data/experiments/0105_rollout \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-type2-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0103 \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 



# python scripts/eval_arx5.py \
#     -i ../Data-Scaling-Laws/data/models/arrange_mouse.ckpt \
#     -o data/experiments/0112_arrange_mouse \
#     --no_mirror --policy_port 8769 --frequency 5 --steps_per_inference 4


# python scripts/eval_arx5.py \
#     -i ../Data-Scaling-Laws/data/models/pour_water.ckpt \
#     -o data/experiments/0112_pour_water \
#     --policy_port 8769 --frequency 5 --steps_per_inference 4

# python scripts/eval_arx5.py \
# -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-lr1e-5-alwayspredact/checkpoints/latest.ckpt \
# -o data/experiments/0109 \
# --different_history_freq \
# --no_mirror --policy_port 8768 

# ------------------------- cup -------------------------------------------------------


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0118_prev_best \
#     --different_history_freq \
#     --no_mirror --policy_port 8768 


# ------------------------- towel -------------------------------------------------------

# python scripts/eval_arx5.py \
#     -i ../Data-Scaling-Laws/data/models/fold_towel.ckpt \
#     -o data/experiments/0118_fold_towel \
#     --policy_port 8769 --frequency 5 --steps_per_inference 4






# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/umi_towel-unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-2/checkpoints/latest.ckpt \
#     -o data/experiments/0105_rollout \
#     --different_history_freq \
#     --policy_port 8768 


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/umi_towel-unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-2-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0118_towel_eval \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4

# ------------------------- mix -------------------------------------------------------
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/umi_mix-unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop/checkpoints/latest.ckpt \
#     -o data/experiments/0116_multi_task_uva \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 8

# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/umi_mix-unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-lr1e-5/checkpoints/latest.ckpt \
#     -o data/experiments/0118_multi_task_uva \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4
    
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/0diffusion_policy_umi_mix/checkpoints/latest.ckpt \
#     -o data/experiments/0116_multi_task \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 8


# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/0diffusion_policy_umi_mix_200data_1000epoch/checkpoints/latest.ckpt \
#     -o data/experiments/0118_mix_dp_200episodes \
#     --no_mirror \
#     --policy_port 8768 --frequency 5 --steps_per_inference 4

python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi/data200_umi_mix-unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-alltask-lr1e-5/checkpoints/latest.ckpt \
    -o data/experiments/0118_mix_200episodes_ours \
    --different_history_freq \
    --no_mirror \
    --policy_port 8768 --frequency 5 --steps_per_inference 4
    
# ------------------------- water -------------------------------------------------------
# python scripts/eval_arx5.py \
#     -i ../diffusion_policy/data/outputs_v4/umi/umi_water-unified-act-autoregressive-cant-proj-newact-conv_fc-newdl-noshiftact-6actlayer-6imagelayer-repcur-normnone-diffhisfreq-no_mirror-diffinterval-newdl2-predprop-3/checkpoints/latest.ckpt \
#     -o data/experiments/0116_pour_water_ours \
#     --different_history_freq \
#     --policy_port 8768 --frequency 5 --steps_per_inference 8