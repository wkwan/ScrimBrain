# reference: https://github.com/DLR-RM/stable-baselines3/issues/1467

import fortnite_env
import os 
import argparse
import atexit
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
parser.add_argument("--framestack_folder_path", type=str, default="media\example-scoring-framestack")
parser.add_argument("--output_path", type=str, default="activations_output")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# for visualizing in real-time during inference
# env = VecFrameStack(make_vec_env(fortnite_env.FortniteEnv, n_envs=1), n_stack=4)
# atexit.register(env.close)
# obs = env.reset()

framestack_img_paths = os.listdir(args.framestack_folder_path)

framestack_imgs = []
for framestack_img_path in framestack_img_paths:
    framestack_imgs.append(Image.open(os.path.join(args.framestack_folder_path, framestack_img_path)))

joined_np_obs = np.concatenate(tuple(framestack_imgs), axis=2)

model = DQN.load(args.checkpoint_path)

# helpful for figuring out which activations to visualize
print(model.policy)

activations_list = []

def get_values(name):
    def hook(model, input, output):
        activations_list.append(output.detach().cpu().numpy()[0])

    return hook

# TODO: which layers are useful to visualize? depends on the model architecture
module_activations = model.policy.q_net.features_extractor.cnn[0]

module_activations.register_forward_hook(get_values(module_activations))

# num_steps = 10

# for i in range(num_steps):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, dones, info = env.step(action)

action, _states = model.predict(joined_np_obs, deterministic=True)

Path(args.output_path).mkdir(exist_ok=True)
plt.ioff()

# save the activation images
for i in range(len(activations_list)):
    for j in range(activations_list[i].shape[0]):
        plt.imshow(activations_list[i][j], cmap='Greys_r')
        plt.axis('off')
        plt.savefig(os.path.join(args.output_path, f"activation_step_{i}_filter_{j}.png"), bbox_inches='tight',transparent=True, pad_inches=0)