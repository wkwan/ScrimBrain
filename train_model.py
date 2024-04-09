import fortnite_env
import os
from pathlib import Path
import argparse
import atexit
from stable_baselines3 import A2C
# from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_folder', type=str, default='checkpoint')
parser.add_argument('--checkpoint_timestep', type=int, default=10240)
parser.add_argument('--use_yolo_reward', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class VecFrameStackSaveOnKill(VecFrameStack):

    def __init__(self, venv, n_stack, starting_timestep=0):
        super().__init__(venv, n_stack)
        self.cur_step = starting_timestep
        self.n_stack = n_stack

    def step_wait(self):
        self.stackedobs, rewards, dones, infos = super().step_wait()
        if (rewards[0] > 0):
            for i in range(self.n_stack):
                Image.fromarray(self.stackedobs[0,:,:,i*3:i*3+3]).save(f"{args.checkpoint_folder}/img_player_killed_opponent_stacked/step_{self.cur_step}_{i}_player_killed_opponent.png")
        elif (rewards[0] < 0):
            for i in range(self.n_stack):
                Image.fromarray(self.stackedobs[0,:,:,i*3:i*3+3]).save(f"{args.checkpoint_folder}/img_opponent_killed_player_stacked/step_{self.cur_step}_{i}_opponent_killed_player.png")
        # else:
        #     for i in range(self.n_stack):
        #         Image.fromarray(self.stackedobs[0,:,:,i*3:i*3+3]).save(f"{args.checkpoint_folder}/stacked/step_{self.cur_step}_{i}.png")
        # print(rewards, dones)
        self.cur_step += 1
        return self.stackedobs, rewards, dones, infos

checkpoint_path = os.path.join(args.checkpoint_folder, f"{args.checkpoint_timestep}.zip")
starting_timestep = 0
if os.path.isfile(checkpoint_path):
    starting_timestep = args.checkpoint_timestep
    env = VecFrameStackSaveOnKill(make_vec_env(fortnite_env.FortniteEnv, n_envs=1, env_kwargs={'use_yolo_reward': args.use_yolo_reward}), n_stack=4, starting_timestep=starting_timestep)
    print("loaded model")
    # model = RecurrentPPO.load(checkpoint_path)
    model = A2C.load(checkpoint_path)
    model.set_env(env)
else:
    print("new model")
    env = VecFrameStackSaveOnKill(make_vec_env(fortnite_env.FortniteEnv, n_envs=1, env_kwargs={'use_yolo_reward': args.use_yolo_reward}), n_stack=4, starting_timestep=0)
    Path(f'{args.checkpoint_folder}/img_player_killed_opponent_stacked').mkdir(parents=True, exist_ok=True)
    Path(f'{args.checkpoint_folder}/img_opponent_killed_player_stacked').mkdir(parents=True, exist_ok=True)
    Path(f'{args.checkpoint_folder}/stacked').mkdir(parents=True, exist_ok=True)
    # model = RecurrentPPO("CnnLstmPolicy", env, n_steps=2048, verbose=1, tensorboard_log=f'{args.checkpoint_folder}/tensorboard')
    model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=f'{args.checkpoint_folder}/tensorboard')

atexit.register(env.close)

for i in range(100):
    model = model.learn(total_timesteps=10000, reset_num_timesteps=False)
    # model = model.learn(total_timesteps=10240, reset_num_timesteps=False)
    # checkpoint_name = f'{args.checkpoint_folder}/{starting_timestep + (10240 * (i+1))}'
    checkpoint_name = f'{args.checkpoint_folder}/{starting_timestep + (10000 * (i+1))}'
    model.save(checkpoint_name)
    # if fortnite_env.has_at_least_one_nonzero_reward_during_learn_phase:
    #     checkpoint_name = f'{args.checkpoint_folder}/{starting_timestep + (10240 * (i+1))}'
    #     model.save(checkpoint_name)
    # fortnite_env.has_at_least_one_nonzero_reward_during_learn_phase = False

