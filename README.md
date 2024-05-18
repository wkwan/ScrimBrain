# ScrimBrain Fortnite Bots

### TLDR: Practice bots in Fortnite could be useful but UEFN doesn't let you control the player or train/run neural nets. This is a hacky demo of 2 use cases for practice bots. If you find this useful, give this a star and hopefully Epic will launch something like the API that allows [RLBot](https://github.com/RLBot/RLBot) to work for Rocket League. The methods used here are a PITA for development.

## Simple Use Case: Wall Holding

This initial motivation for this project was to develop a bot to hold a wall in Fortnite, so players can practice the wall stealing mechanic without a practice partner. Because Fortnite doesn't have an API to control the player, we did this by simulating Windows keypresses in Python. 

The very simple script to hold the wall (or other build piece) is [hold_wall.py](hold_wall.py). It requires manual tweaking but it works for basic practice. 

Instructions:
1. Run Fortnite on a second PC.
2. Change the keybind in [hold_wall.py](hold_wall.py) from 'o' to whatever your keybind is for "Place Build".
3. Walk the player on your second PC behind the wall you want to hold and take out your wall.
4. From there, running the script simply holds the keybind to place the wall and you can practice the wall stealing mechanic on your main PC.

Here's what this looks like for the wall stealing player. The bot is in the box holding the wall.
![](https://github.com/wkwan/ScrimBrain/blob/master/media/wall-steal.gif)

You can use our map **Steal My Wall!** to practice 7 different build piece stealing scenarios: https://www.fortnite.com/@coachdody/2191-1425-4724 

## Complex Use Case: 1v1 Zerobuild Fight

Since we can simulate Windows inputs, and also ingest real-time screencapture of Fortnite, we can train reinforcement learning agents for Fortnite. This is like how [OpenAI Universe](https://github.com/openai/universe) and [SerpentAI](https://github.com/SerpentAI/SerpentAI) worked. The problem is that it requires more training and bigger models compared to using an MLP that takes useful game state features like player positions, inventory, etc. as inputs. The benefit is that it could be extended to support almost any PC game.

The current debugging model is trained on this map: https://www.fortnite.com/@necrogames/8136-5511-4930 

Checkpoint: https://drive.google.com/file/d/1YQMV1YDcrUWauTdafpmKbVrT74jLi9Pn/view?usp=sharing

Demo: https://youtu.be/OnpnFiNthDA?feature=shared&t=39

It should work best using the same map and skins shown in the demo, but it isn't very smart regardless and training for more timesteps doesn't seem to help. One possible reason is that the screencapture is too complex for the model to learn. Another possible reason is the training setup. We trained 2 neural nets simultaneously (one for the chicken and one for the default blonde female character) so that we could generate infinite 1v1 gameplay training data that improves (in terms of gameplay level) as the models improve. But the random initialization might make the initial training data quality too low, so the model converges too early.

### Run the Pretrained Model

Setup your keybinds to match the training environment keybinds in [fortnite_env.py](fortnite_env.py)

Fullscreen Fortnite at 1920x1080 on your main monitor.

Load into the map (**1V1, 1 HP, 1X1 BOX** 8136-5511-4930).

```
conda env create -f environment.yml
conda activate scrimbrain
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
python run_model.py --checkpoint_path=YOUR_CHECKPOINT_PATH
```

### Train Your Own Model
If you really want to try, you can open an issue and [@wkwan](https://www.github.com/wkwan) will try to help.
