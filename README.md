# ScrimBrain - Reinforcement Learning Agents for Fortnite 1v1's

ScrimBrain aims to help competitive gamers of all skill levels improve more efficiently. To do this, we’re building an open-source reinforcement learning framework to train practice bots for games, starting with Fortnite 1v1’s. 

The current model is trained on this map: https://www.fortnite.com/@necrogames/8136-5511-4930

The model ingests real-time screen capture and predicts keyboard/mouse inputs to simulate. Although Fortnite is the first use-case, the system is designed to work with any game that runs on Windows.

## Setup

```
conda env create -f environment.yml
conda activate scrimbrain
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### TODO: model checkpoints, more documentation

 
