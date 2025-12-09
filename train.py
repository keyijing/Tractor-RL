from replay_buffer import ReplayBuffer
from learner import Learner
from rollout import Actor
from agent import N_TOKENS, N_ACTIONS
import os
import multiprocessing as mp

config = {
	'gamma': 0.98,
	'lambda': 0.95,
	'log': 'wandb',
	'rl': {
		'eps': 0.2,
		'value_coef': 0.5,
		'entropy_coef': 1e-3,
		'aux_coef': 0.1,
		'batch_size': 2048,
		'mini_batch_size': 64,
		'epochs': 3,
		'clip_grad': 1,
		'ckpt_save_interval': 100,
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
		'd_model': 256,
		'max_seq_len': 384,
		'num_blocks': 16,
		'num_heads': 8,
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
	actors = [Actor(i, [2, 3, 4, 5], dataset, config) for i in range(config['actor']['n_actors'])]

	for learner in learners:
		learner.start()
	for actor in actors:
		actor.start()
	for learner in learners:
		learner.join()
	for actor in actors:
		actor.join()
