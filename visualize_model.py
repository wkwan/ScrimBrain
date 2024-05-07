import fortnite_env
import os 
import argparse
import atexit
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
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

# TODO: Which layers to visualize?
# module_activations = model.policy.pi_features_extractor.cnn[0]
module_activations = model.policy.features_extractor.cnn[0]
# module_activations = model.policy.pi_features_extractor.linear[0]
# module_activations = model.policy.mlp_extractor.policy_net[0]

module_activations.register_forward_hook(get_values(module_activations))

num_steps = 10

for i in range(num_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)

for i in range(len(activations_list)):
    # TODO: do we need to normalize?
    # activations_list[i] -= activations_list[i].min()
    # activations_list[i] /= activations_list[i].max()

    plt.figure(figsize=(8, 8))
    plt.imshow(activations_list[i][0]) # TODO: confirm if this is the correct way to visualize the CNN filters. If it is, then visualize all of them in a grid.
    print(activations_list[i][0])
    plt.axis('off')
    plt.show()
