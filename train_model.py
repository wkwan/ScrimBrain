import fortnite_env
import os
from pathlib import Path
import argparse
import atexit
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_folder', type=str, default='checkpoint')
parser.add_argument('--checkpoint_timestep', type=int, default=10240)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class VecFrameStackSaveOnKill(VecFrameStack):
    def __init__(self, venv, n_stack, starting_timestep=0):
        super().__init__(venv, n_stack)
        self.cur_step = starting_timestep
        self.n_stack = n_stack

    def step_wait(self):
        self.stackedobs, rewards, dones, infos = super().step_wait()
        if (rewards[0] > 0): # when there's a reward, save the images for reference
            for i in range(self.n_stack):
                Image.fromarray(self.stackedobs[0,:,:,i*3:i*3+3]).save(f"{args.checkpoint_folder}/img_score/step_{self.cur_step}_{i}_score.png")
        self.cur_step += 1
        return self.stackedobs, rewards, dones, infos

checkpoint_path = os.path.join(args.checkpoint_folder, f"{args.checkpoint_timestep}.zip")
starting_timestep = 0
env = None
if os.path.isfile(checkpoint_path):
    starting_timestep = args.checkpoint_timestep
    env = VecFrameStackSaveOnKill(make_vec_env(fortnite_env.FortniteEnv, n_envs=1, n_stack=4, starting_timestep=starting_timestep)
    print("loaded model")
    model = DQN.load(checkpoint_path)
    model.set_env(env)
else:
    print("new model")
    env = VecFrameStackSaveOnKill(make_vec_env(fortnite_env.FortniteEnv, n_envs=1, n_stack=4, starting_timestep=0)
    Path(f'{args.checkpoint_folder}/img_score').mkdir(parents=True, exist_ok=True)
    model = DQN("CnnPolicy", env, buffer_size=20000, verbose=1, tensorboard_log=f'{args.checkpoint_folder}/tensorboard')

atexit.register(env.close)

for i in range(1000):
    model = model.learn(total_timesteps=50000, reset_num_timesteps=False)
    checkpoint_name = f'{args.checkpoint_folder}/{starting_timestep + (50000 * (i+1))}'
    model.save(checkpoint_name)