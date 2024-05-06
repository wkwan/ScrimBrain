# ScrimBrain Fortnite Bots

## Origin

This initial motivation for this project was to develop a bot to hold a wall in Fortnite, so players can practice the wall-stealing mechanic without a practice partner. Because Fortnite doesn't have an API to control the player, we did this by simulating Windows keypresses in Python. 

The very simple script to hold the wall (or other build piece) is [hold_wall.py](hold_wall.py). It requires some manual tweaks but it works for basic practice. The idea is to run Fortnite on a second PC, setup the appropriate keybind for placing the build, and then manually position the player on your second PC behind the wall you want to hold and take out your wall. From there, running the script simply holds the keybind to place the wall and you can practice the wall-stealing mechanic on your main PC.

Here's what this looks like for the wall-stealing player. The bot is in the box holding the wall.
![](https://github.com/wkwan/ScrimBrain/blob/master/media/wall-steal.gif)

We'll soon be launching a Fortnite map that can optionally be used with the script to practice wall-stealing.

## Vision

Since we can simulate Windows inputs, and also ingest real-time screencapture of Fortnite, we can train reinforcement learning agents for Fortnite. This is like how [OpenAI Universe](https://github.com/openai/universe) and [SerpentAI](https://github.com/SerpentAI/SerpentAI) worked.

ScrimBrain aims to help competitive gamers of all skill levels improve more efficiently. To do this, we’re building an open-source reinforcement learning framework to train practice bots for games, starting with Fortnite 1v1’s. This is the first reinforcement learning framework targeted to gamers, as opposed to AI researchers.

The current model is trained on this map: https://www.fortnite.com/@necrogames/8136-5511-4930

The model ingests real-time screen capture and predicts keyboard/mouse inputs to simulate. 

Soon, we’ll be offering model checkpoints trained on different UEFN maps and performing at different skill levels to suit various practice needs.

Although Fortnite is the first use-case, the system is designed to work with any game that runs on Windows, and we'll be adding extensive documentation to help developers train their own models.

## Setup

```
conda env create -f environment.yml
conda activate scrimbrain
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### TODO: model checkpoints, more documentation

 
