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
obs = env.reset()

model = RecurrentPPO.load(args.checkpoint_path)

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)
