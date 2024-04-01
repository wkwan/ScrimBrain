import fortnite_env
import os
import argparse
import atexit
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env = VecFrameStack(make_vec_env(fortnite_env.FortniteEnv, n_envs=1), n_stack=4)
atexit.register(env.close)

if os.path.isfile(args.checkpoint_path):
    model = RecurrentPPO.load(args.checkpoint_path)
    model.set_env(env)
else:
    model = RecurrentPPO("CnnLstmPolicy", env, n_steps=2048, verbose=1, tensorboard_log=f'{args.checkpoint_path}_tb')

for i in range(100):
    model = model.learn(total_timesteps=10240, reset_num_timesteps=False)
    if fortnite_env.has_at_least_one_nonzero_reward_during_learn_phase:
        checkpoint_name = f'{args.checkpoint_path}_{10240 * (i+1)}'
        model.save(checkpoint_name)
    fortnite_env.has_at_least_one_nonzero_reward_during_learn_phase = False

