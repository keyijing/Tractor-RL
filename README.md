# Generative RL for Tractor (双升游戏 AI)

This is the term project for Reinforcement Learning course in Peking University (2025 Fall). This project presents a generative reinforcement learning approach for the card game Tractor ([双升](https://www.botzone.org.cn/game/Tractor)).

## Project Report

See the [project report](report.pdf) for the details.

## Building and Running

### Requirements

- Python 3.11
- Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Setup

Before training, you need to compile the C++ extension module:

```bash
python setup.py build_ext --inplace
```

This command compiles `tractor.cpp` and creates the necessary Python bindings.

### Training

To start training the model, run the following command:

```bash
python train.py
```

## Project Structure

```
.
├── tractor.cpp           # C++ implementation of the game engine
├── train.py              # Main training script
├── model.py              # Neural network model definition
├── agent.py              # Agent implementation
├── env.py                # Environment wrapper
├── game.py               # Game logic
├── learner.py            # Learner process for training
├── rollout.py            # Actor process for rollout
├── replay_buffer.py      # Experience replay buffer
└── model_pool.py         # Model pool for self-play
```

