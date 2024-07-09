# Reinforcement Learning in Fortnite With Real-Time Screen Capture and Windows Input Simulation

### Motivation: How do you train a reinforcement learning agent to play a game without special access to the game like source code access, API access, or RAM hacking? And can you do it with 1 GPU?

Most games don't have API access and most developers don't have access to GPU farms. If we can train AI to play Fortnite on a single PC, we can train AI to play almost any game on a single PC. This could be useful as practice tools for competitive players, and or for automated playtesting (finding bugs, game balancing, etc.).

ScrimBrain works similarly to how [OpenAI Universe](https://github.com/openai/universe) and [SerpentAI](https://github.com/SerpentAI/SerpentAI) worked. The challenge with using screencapture is that it requires more training and bigger models compared to using an MLP that takes useful game state features like player positions, inventory, etc. as inputs. It also makes it harder to write a reward function. The benefit is that it could be extended to support almost any PC game.

## Run the example reinforcement learning agent trained on a custom Fortnite map

![](https://github.com/wkwan/ScrimBrain/blob/master/media/scrimbrain-race-example.gif)

In this map, the goal is to run the target as fast as possible. The model was trained by getting a reward by detecting the blue **SCORE** text that appears on the screen when reaching the target.

1. Clone this repo
2. Download the checkpoint: https://drive.google.com/file/d/1xly-4-C2f_PWK2g_3Ks18AbdkUllRpyV/view?usp=sharing
3. Setup the conda environment:
```
conda env create -f environment.yml
conda activate scrimbrain
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
4. Set your main monitor resolution to 1920x1080 (and 60Hz if possible) to match the training setup. Fullscreen Fortnite on your main monitor. The Python script will take screencaptures at this resolution on your main monitor. The full-sized screencapture will be used for the reward function, while the neural net will see a downscaled screencapture.
5. Set these game settings and leave the rest to default:  
```
Window Mode: Fullscreen  
Resolution: 1920 x 1080  
Vsync: ON  
Frame Rate Limit: 60 FPS  
Rendering Mode: Performance - Lower Graphical Fidelity  
3D Resolution: 0%  
View Distance: Epic  
Textures: Low  
Meshes: Low  
X-Axis Sensitivity: 3.8%  
Move Forward: W  
HUD Options: HUD Scale 125%, turn everything else off
```
6. Load into the map **ScrimBrain Race Example** (map code 7340-6949-212). Don't move the cursor, the model can only look left/right and not up/down, so if you move the cursor up/down, you'll be showing the model a different perspective than what it was trained on, and it won't be able to correct it during inference. Note that the framerate of the model isn't rate-limited, so it depends on how fast your PC can chug through frames and run inference. It's possible that this will cause reproducibility issues.
7. Run the model. If your terminal is on another monitor, make sure to move the cursor to the main Fortnite monitor after running the script, click the screen to set the focus to the main monitor, and don't move the cursor afterwards.
```
python run_model.py --checkpoint_path=YOUR_CHECKPOINT_PATH
```

## Train the example model

Follow Steps 1-6 above.

To train from scratch:
```
python train_model.py --checkpoint_folder=YOUR_CKPT_FOLDER 
```

To train from a checkpoint:
```
python train_model.py --checkpoint_folder=YOUR_CKPT_FOLDER --checkpoint_timestep=TIMESTEP_INT
```

## Visualize the convolutional filters in a pretrained DQN model

```
python visualize_model.py --checkpoint_path=PATH_TO_DQN_MODEL --framestack_folder_path=PATH_TO_FOLDER_WITH_4_FRAMESTCK_IMGS --output_path=OUTPUT_PATH
```

Modify the **module_activations** variable in [visualize_model.py](visualize_model.py) to visualize different activations.

[visualize_model.py](visualize_model.py) can be easily adapted to different architectures with different activations.

## Misc

Sometimes you don't need AI, macros are good enough.

A simple Fortnite use case for macros that isn't cheating is holding a wall. This lets players can practice the wall stealing mechanic without a practice partner.

The very simple script to hold the wall (or other build piece) is [hold_wall.py](hold_wall.py). It requires manual tweaking but it works for basic practice. 

Instructions:
1. Run Fortnite on a second PC.
2. Change the keybind in [hold_wall.py](hold_wall.py) from 'o' to whatever your keybind is for "Place Build".
3. Walk the player on your second PC behind the wall you want to hold and take out your wall.
4. From there, running the script simply holds the keybind to place the wall and you can practice the wall stealing mechanic on your main PC.

Here's what this looks like for the wall stealing player. The bot is in the box holding the wall.
![](https://github.com/wkwan/ScrimBrain/blob/master/media/wall-steal.gif)

You can use our map **Steal My Wall!** to practice 7 different build piece stealing scenarios: https://www.fortnite.com/@coachdody/2191-1425-4724 

In addition to the guided practice areas, **Steal My Wall!** has a 1v1 competitive mode focused on stealing pieces, taking good peeks, and strategically using exploits to get into your opponent’s box.

