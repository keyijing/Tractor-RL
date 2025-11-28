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
	'sl': {
		'device': 'cuda:0',
		'model_pool_size': 4,
		'batch_size': 1024,
		'mini_batch_size': 64,
		'epochs': 1,
		'clip_grad': 1,
		'ckpt_save_interval': 50,
		'ckpt_save_path': 'checkpoint/model_avg',
		'replay_buffer': {
			'capacity': 16384,
			'episode': 64,
			'seed': 0,
		},
		'optim': {
			'lr': 1e-5,
			'weight_decay': 1e-2,
		},
	},
	'rl': {
		'device': 'cuda:7',
		'model_pool_size': 4,
		'eps': 0.2,
		'value_coef': 0.5,
		'entropy_coef': 0.005,
		'batch_size': 1024,
		'mini_batch_size': 64,
		'epochs': 3,
		'clip_grad': 1,
		'ckpt_save_interval': 50,
		'ckpt_save_path': 'checkpoint/model_best',
		'replay_buffer': {
			'capacity': 1536,
			'episode': 64,
			'seed': 0,
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
	os.makedirs(config['sl']['ckpt_save_path'], exist_ok=True)
	os.makedirs(config['rl']['ckpt_save_path'], exist_ok=True)

	datasets = {
		'avg': ReplayBuffer(**config['sl']['replay_buffer']),
		'best': ReplayBuffer(**config['rl']['replay_buffer']),
	}
	learner = Learner(datasets, config)
	actors = [Actor(i + 2, datasets, config) for i in range(config['actor']['n_actors'])]

	learner.start()
	for actor in actors:
		actor.start()
	learner.join()
	for actor in actors:
		actor.join()
