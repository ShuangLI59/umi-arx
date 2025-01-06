import sys
import os

from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import zmq

@click.command()
@click.option("--ckpt_path", type=str, required=True)
@click.option("--policy_ip", type=str, required=True)
@click.option("--policy_port", type=int, required=True)
@click.option("--data_dir", type=str, required=True)
def main(ckpt_path: str, policy_ip: str, policy_port: int, data_dir: str):
    obs_dir = os.path.join(data_dir, "obs")
    new_action_dir = os.path.join(data_dir, "new_action")
    os.makedirs(new_action_dir, exist_ok=True)

    cfg_path = ckpt_path.replace(".ckpt", ".yaml")
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.load(f)

    obs_pose_repr = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{policy_ip}:{policy_port}")

    for episode_obs in os.listdir(obs_dir):
        episode_obs_path = os.path.join(obs_dir, episode_obs)
        for obs_file in os.listdir(episode_obs_path):
            obs_file_path = os.path.join(episode_obs_path, obs_file)
            with open(obs_file_path, "rb") as f:
                obs = pickle.load(f)


if __name__ == "__main__":
    main()
    # data/outputs_v4/umi/unified-act-autoregressive-cant-proj-proprioception-newact-newdl/checkpoints/latest.ckpt