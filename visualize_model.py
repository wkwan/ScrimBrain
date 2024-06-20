# reference: https://github.com/DLR-RM/stable-baselines3/issues/1467

import fortnite_env
import os 
import argparse
import atexit
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env = VecFrameStack(make_vec_env(fortnite_env.FortniteEnv, n_envs=1), n_stack=4)
atexit.register(env.close)
obs = env.reset()

model = A2C.load(args.checkpoint_path)

activations_list = []

def get_values(name):
    def hook(model, input, output):
        activations_list.append(output.detach().cpu().numpy()[0])

    return hook

# TODO: which layers are useful to visualize? probably depends on the model
# module_activations = model.policy.pi_features_extractor.cnn[0]
module_activations = model.policy.features_extractor.cnn[0]
# module_activations = model.policy.pi_features_extractor.linear[0]
# module_activations = model.policy.mlp_extractor.policy_net[0]

module_activations.register_forward_hook(get_values(module_activations))

num_steps = 10

for i in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)

Path("activations").mkdir(exist_ok=True)
plt.ioff()

# save the activation images
for i in range(len(activations_list)):
    for j in range(activations_list[i].shape[0]):
        print(activations_list[i][j].shape)
        plt.imshow(activations_list[i][j], cmap='Greys_r')
        plt.axis('off')
        plt.savefig(f'activations/activation_step_{i}_filter_{j}.png', bbox_inches='tight',transparent=True, pad_inches=0)