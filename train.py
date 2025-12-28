from replay_buffer import ReplayBuffer
from learner import Learner
from rollout import Actor
from agent import N_TOKENS, N_ACTIONS
import os
import multiprocessing as mp

config = {
	'gamma': 0.99,
	'lambda': 0.95,
	'log': 'wandb',
	'reward_coef': {
		'reward': 0.05,
		'punish': 0.0,
		'final': 0.0,
	},
	'rl': {
		'eps': 0.2,
		'value_coef': 0.5,
		'entropy_coef': 5e-3,
		'aux_coef': 0.05,
		'batch_size': 1024,
		'mini_batch_size': 64,
		'epochs': 1,
		'clip_grad': 1,
		'ckpt_save_interval': 200,
		'ckpt_save_path': 'checkpoint',
		'n_learners': 2,
		'replay_buffer': {
			'capacity': 1536,
			'episode': 64,
			'seed': 0,
		},
		'model_pool': {
			'best': {
				'size': 2,
			},
			'ckpt': {
				'size': 16,
			},
		},
		'optim': {
			'lr': 1e-5,
			'eps': 1e-5,
			'weight_decay': 0,
		}
	},
	'model': {
		'n_toks': N_TOKENS,
		'n_players': 4,
		'n_actions': N_ACTIONS,
		'd_model': 384,
		'max_seq_len': 320,
		'num_blocks': 16,
		'num_heads': 12,
	},
	'actor': {
		'n_actors': 4,
		'batch_size': 128,
		'seed': 42,
	},
}

if __name__ == '__main__':

	mp.set_start_method('spawn')
	os.makedirs(config['rl']['ckpt_save_path'], exist_ok=True)

	dataset = ReplayBuffer(**config['rl']['replay_buffer'])
	learners = [Learner(i, [6, 7], dataset, config) for i in range(config['rl']['n_learners'])]
	actors = [Actor(i, [0, 2, 3, 4], dataset, config) for i in range(config['actor']['n_actors'])]

	for learner in learners:
		learner.start()
	for actor in actors:
		actor.start()
	try:
		for learner in learners:
			learner.join()
		for actor in actors:
			actor.join()
	finally:
		for learner in learners:
			if learner.is_alive():
				learner.kill()
		for actor in actors:
			if actor.is_alive():
				actor.kill()
