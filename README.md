# Reinforcement Learning in Fortnite With Real-Time Screencapture and Windows Input Simulation

### Motivation: How do you train a reinforcement learning agent to play a game without special access to the game like source code access, API access, or RAM hacking? And can you do it with 1 GPU?

Most games don't have API access and most developers don't have access to GPU farms. If we can train AI to play Fortnite on a single PC, we can train AI to play almost any game on a single PC. This could be useful as practice tools for competitive players, and or for automated playtesting (finding bugs, game balancing, etc.).

ScrimBrain works similarly to how [OpenAI Universe](https://github.com/openai/universe) and [SerpentAI](https://github.com/SerpentAI/SerpentAI) worked. The challenge with using screencapture is that it requires more training and bigger models compared to using an MLP that takes useful game state features like player positions, inventory, etc. as inputs. It also makes it harder to write a reward function. The benefit is that it could be extended to support almost any PC game.

## Run the example reinforcement learning agent trained on a custom Fortnite map

_TODO add demo gif_

In this map, the goal is to run the target as fast as possible. The model was trained by getting a reward by detecting the blue **SCORE** text that appears on the screen when reaching the target.

1. Clone this repo
2. Download the checkpoint: _TODO share ckpt_
3. Setup the conda environment
```
conda env create -f environment.yml
conda activate scrimbrain
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
4. Fullscreen Fortnite at 1920x1080 on your main monitor. The Python script will take screencaptures at this resolution on your main monitor. The full-sized screencapture will be used for the reward function, while the neural net will see a downscaled screencapture.
5. Setup your keybinds to match the training environment keybinds in [fortnite_env.py](fortnite_env.py)
6. Load into the map **ScrimBrain Race Example** (map code 7340-6949-212). Don't move the cursor, the model can only look left/right and not up/down, so if you move the cursor up/down, you'll be showing the model a different perspective than what it was trained on, and it won't be able to correct it during inference.
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

