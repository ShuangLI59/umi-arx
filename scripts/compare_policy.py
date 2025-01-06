import sys
import os
from queue import Queue
from omegaconf import open_dict

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import pickle
import numpy as np
import zmq
import matplotlib.pyplot as plt
import cv2
@click.command()
@click.option("--policy_port", type=int, required=True)
@click.option("--policy_ip", type=str, default="127.0.0.1")
@click.option("--obs_dir", type=str, required=True)
@click.option("--action_dir", type=str, required=True)
def main(policy_port: int, policy_ip: str, obs_dir: str, action_dir: str):

    socket = zmq.Context().socket(zmq.REQ)
    socket.connect(f"tcp://{policy_ip}:{policy_port}")

    for obs_file in os.listdir(obs_dir)[1:]:
        if not obs_file.endswith("npy"):
            continue

        obs_file_path = os.path.join(obs_dir, obs_file)
        action_file_path = os.path.join(action_dir, obs_file)
        with open(obs_file_path, "rb") as f:
            obs = np.load(f, allow_pickle=True).item()
        with open(action_file_path, "rb") as f:
            action = np.load(f, allow_pickle=True).item()
        # print(obs.keys())
        # print(action.keys())

        raw_action = action["raw_action"]
        raw_action_xyz = raw_action[:, :3]
        raw_obs_xyz = obs["obs_dict_np"]["robot0_eef_pos"][:, :3].copy()


        cv2_img = np.concatenate([img*255 for img in obs["obs_dict_np"]["camera0_rgb"]], axis=2).swapaxes(0, 2)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2_img = cv2_img.astype(np.uint8).swapaxes(0, 1)
        # cv2.imshow("obs", cv2_img)
        # cv2.waitKey(0)
        cv2.imwrite(f"{obs_dir}/{obs_file.split('.')[0]}_img.png", cv2_img)
        

        obs_dict_np = obs["obs_dict_np"]
        for key, val in obs_dict_np.items():
            obs_dict_np[key] = val[-2:]
            # print(key, val.shape)
            # if len(val.shape) == 2:
            #     print(val)
        # print(obs_dict_np["camera0_rgb"].shape)
        # cv2.imshow("obs", obs["original_imgs"][-1])
        obs_dict_np["camera0_rgb"] = obs["original_imgs"][-1:].swapaxes(3, 1)
        print(obs_dict_np["camera0_rgb"].shape) 
        obs_pose_rep = obs["obs_pose_rep"]

        socket.send_pyobj(obs_dict_np)
        new_action = socket.recv_pyobj()
        if isinstance(new_action, str):
            print(new_action)
            break
        new_action_xyz = new_action[:, :3]

        # Plot raw_action_xyz and new_action_xyz
        plt.plot(raw_action_xyz[:, 0], 'r-', label='ours_x')
        plt.plot(new_action_xyz[:, 0], 'r.', label='dp_x')
        plt.plot(raw_action_xyz[:, 1], 'g-', label='ours_y')
        plt.plot(new_action_xyz[:, 1], 'g.', label='dp_y')
        plt.plot(raw_action_xyz[:, 2], 'b-', label='ours_z')
        plt.plot(new_action_xyz[:, 2], 'b.', label='dp_z')
        plt.legend()
        plt.savefig(f"{obs_dir}/{obs_file.split('.')[0]}_actions.png")
        plt.close()

        # Plot raw_obs_xyz
        plt.plot(raw_obs_xyz[:, 0], 'r-', label='raw_obs_x')
        plt.plot(raw_obs_xyz[:, 1], 'g-', label='raw_obs_y')
        plt.plot(raw_obs_xyz[:, 2], 'b-', label='raw_obs_z')
        plt.legend()
        plt.savefig(f"{obs_dir}/{obs_file.split('.')[0]}_obs.png")
        plt.close()

        

if __name__ == "__main__":
    main()