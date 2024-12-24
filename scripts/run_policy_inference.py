import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click


@click.command()
@click.option("--ckpt_path", type=str, required=True)
@click.option("--policy_ip", type=str, required=True)
@click.option("--policy_port", type=int, required=True)
@click.option("--data_dir", type=str, required=True)
def main(ckpt_path: str, policy_ip: str, policy_port: int, data_dir: str):
    obs_dir = os.path.join(data_dir, "obs")
    new_action_dir = os.path.join(data_dir, "new_action")
    os.makedirs(new_action_dir, exist_ok=True)

    ckpt_path = 


if __name__ == "__main__":
    main()