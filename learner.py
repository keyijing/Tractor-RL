from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import Model
from multiprocessing import Process
from pathlib import Path
import wandb
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader

def entropy(logits: torch.Tensor):
	"""
	Assume logits are actually log_probs (log_softmax output)
	"""
	probs = torch.exp(logits)
	# Mask out 0 probabilities to avoid NaN from 0*-inf
	logits = torch.where(probs == 0, 0, logits)
	p_log_p = probs * logits
	return -p_log_p.sum(dim=-1)

def KL_divergence(logits: torch.Tensor, old_logits: torch.Tensor):
	"""
	Assume logits and old_logits are actually log_probs (log_softmax output)
	"""
	probs = torch.exp(logits)
	# Mask out 0 probabilities to avoid NaN from inf-inf
	logits = torch.where(probs == 0, 0, logits - old_logits)
	return (probs * logits).sum(dim=-1)

def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
	logits = logits.log_softmax(dim=-1)
	# Mask out 0 probabilities to void Nan from 0*-inf
	logits = torch.where(target == 0, 0, logits)
	return -(target * logits).sum(dim=-1)

def masked_normalize(advantages: torch.Tensor, output_mask: torch.Tensor, eps = 1e-5):
	mean = advantages[output_mask].mean()
	std = advantages[output_mask].std() + eps
	return (advantages - mean) / std

class SLLearner:
	def __init__(self, name: str, models: dict[str, Model], replay_buffers: dict[str, ReplayBuffer], config: dict):
		self.name = name
		self.config = config
		self.device = config['device']
		self.model = models[name].to(self.device)
		self.replay_buffer = replay_buffers[name]
		self.model_pool = ModelPoolServer(config['model_pool_size'], name)
		self.model_pool.push(self.model.state_dict())
		self.optimizer = torch.optim.AdamW(self.model.parameters(), **config['optim'])
		self.iterations = 0

	def step(self):
		batch_size = self.config['batch_size']
		mini_batch_size = self.config['mini_batch_size']

		# sample batch
		batch = self.replay_buffer.sample(batch_size)
		toks = torch.tensor(batch['toks'], device=self.device)
		output_masks = torch.tensor(batch['output_mask'], device=self.device)
		logits = torch.tensor(batch['logits'], device=self.device)
		# print(f'{toks.shape=}')
		# print(f'{output_masks.shape=}')
		# print(f'{logits.shape=}')

		print(f'SL Iteration {self.iterations}, replay buffer in {self.replay_buffer.stats["sample_in"]} out {self.replay_buffer.stats["sample_out"]}')
		self.model.train(True) # Training mode

		dataset = TensorDataset(toks, output_masks, logits)
		dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

		stats = {
			'loss': []
		}

		for _ in range(self.config['epochs']):
			for tok, output_mask, old_logit in dataloader:
				total = output_mask.sum()
				def _masked_mean(x: torch.Tensor):
					return torch.where(output_mask, x, 0).sum() / total

				action_mask = old_logit != -torch.inf
				target = F.softmax(old_logit, dim=-1)
				logit, _ = self.model(tok[..., 0], tok[..., 1])
				logit = torch.where(action_mask, logit, -torch.inf)
				loss = _masked_mean(cross_entropy(logit, target))

				stats['loss'].append(loss.item())

				self.optimizer.zero_grad()
				loss.backward()
				if 'clip_grad' in self.config:
					clip_grad_norm_(self.model.parameters(), max_norm=self.config['clip_grad'])
				self.optimizer.step()

		# push new model
		self.model_pool.push(self.model.state_dict())
		self.iterations += 1
		if self.iterations % self.config['ckpt_save_interval'] == 0:
			path = Path(self.config['ckpt_save_path'], f'{self.iterations}.pt')
			torch.save(self.model.state_dict(), path)

		return {
			key: sum(value) / len(value) for key, value in stats.items()
		}

class RLLearner:
	def __init__(self, name: str, models: dict[str, Model], replay_buffers: dict[str, ReplayBuffer], config: dict):
		self.name = name
		self.config = config
		self.device = config['device']
		self.model = models[name].to(self.device)
		self.replay_buffer = replay_buffers[name]
		self.model_pool = ModelPoolServer(config['model_pool_size'], name)
		self.model_pool.push(self.model.state_dict())
		self.optimizer = torch.optim.AdamW(self.model.parameters(), **config['optim'])
		self.iterations = 0

	def step(self):
		eps = self.config['eps']
		value_coef = self.config['value_coef']
		entropy_coef = self.config['entropy_coef']
		batch_size = self.config['batch_size']
		mini_batch_size = self.config['mini_batch_size']

		# sample batch
		batch = self.replay_buffer.pop(batch_size)
		toks = torch.tensor(batch['toks'], device=self.device)
		actions = torch.tensor(batch['actions'], device=self.device)
		output_masks = torch.tensor(batch['output_mask'], device=self.device)
		log_probs = torch.tensor(batch['log_probs'], device=self.device)
		# Assume logits are actually log_probs (log_softmax output)
		logits = torch.tensor(batch['logits'], device=self.device)
		targets = torch.tensor(batch['values'], device=self.device)
		advantages = torch.tensor(batch['advantages'], device=self.device)
		advantages = masked_normalize(advantages, output_masks)
		reward = np.mean(np.where(batch['output_mask'], batch['rewards'], 0).sum(axis=-1)).item()
		# print(f'{toks.shape=}')
		# print(f'{actions.shape=}')
		# print(f'{output_masks.shape=}')
		# print(f'{log_probs.shape=}')
		# print(f'{logits.shape=}')
		# print(f'{targets.shape=}')
		# print(f'{advantages.shape=}')

		print(f'RL Iteration {self.iterations}, replay buffer in {self.replay_buffer.stats["sample_in"]} out {self.replay_buffer.stats["sample_out"]}')
		self.model.train(True) # Training mode

		dataset = TensorDataset(toks, actions, output_masks, log_probs, logits, targets, advantages)
		dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

		stats = {
			'reward': [reward],
			'policy_loss': [],
			'value_loss': [],
			'entropy_loss': [],
			'total_loss': [],
			'kl_div': [],
		}

		for _ in range(self.config['epochs']):
			for tok, action, output_mask, old_log_prob, old_logit, target, adv in dataloader:
				total = output_mask.sum()
				def _masked_mean(x: torch.Tensor):
					return torch.where(output_mask, x, 0).sum() / total

				action_mask = old_logit != -torch.inf
				logit, value = self.model(tok[..., 0], tok[..., 1])
				logit = torch.where(action_mask, logit, -torch.inf)
				logit = F.log_softmax(logit, dim=-1)
				log_prob = logit.gather(-1, action.unsqueeze(dim=-1)).squeeze(dim=-1)
				ratio = torch.exp(log_prob - old_log_prob)
				surr1 = ratio * adv
				surr2 = torch.clip(ratio, 1 - eps, 1 + eps) * adv
				policy_loss = -_masked_mean(torch.min(surr1, surr2))
				value_loss = _masked_mean(F.mse_loss(value, target, reduction='none'))
				entropy_loss = -_masked_mean(entropy(logit))
				KL_div = _masked_mean(KL_divergence(logit, old_logit))
				loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

				stats['policy_loss'].append(policy_loss.item())
				stats['value_loss'].append(value_loss.item())
				stats['entropy_loss'].append(entropy_loss.item())
				stats['total_loss'].append(loss.item())
				stats['kl_div'].append(KL_div.item())

				self.optimizer.zero_grad()
				loss.backward()
				if 'clip_grad' in self.config:
					clip_grad_norm_(self.model.parameters(), max_norm=self.config['clip_grad'])
				self.optimizer.step()

		# push new model
		self.model_pool.push(self.model.state_dict())
		self.iterations += 1
		if self.iterations % self.config['ckpt_save_interval'] == 0:
			path = Path(self.config['ckpt_save_path'], f'{self.iterations}.pt')
			torch.save(self.model.state_dict(), path)

		return {
			key: sum(value) / len(value) for key, value in stats.items()
		}

class Learner(Process):
	def __init__(self, replay_buffers: dict[str, ReplayBuffer], config: dict):
		super(Learner, self).__init__()
		self.replay_buffers = replay_buffers
		self.config = config

	def _flush(self):
		while True:
			self.sl_learner.replay_buffer._flush()
			self.rl_learner.replay_buffer._flush()
			time.sleep(0.1)

	def run(self):
		if self.config.get('log') == 'wandb':
			time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
			wandb.init(project='Tractor-RL', name=f'Learner-{time}', config=self.config)

		models = {
			name: Model(**self.config['model'])
			for name in self.replay_buffers
		}
		self.sl_learner = SLLearner('avg', models, self.replay_buffers, self.config['sl'])
		self.rl_learner = RLLearner('best', models, self.replay_buffers, self.config['rl'])

		thread = threading.Thread(target=self._flush, daemon=True)
		thread.start()
		try:
			while True:
				with ThreadPoolExecutor(max_workers=2) as exec:
					sl = exec.submit(self.sl_learner.step)
					rl = exec.submit(self.rl_learner.step)
					sl_stats = sl.result()
					rl_stats = rl.result()

				if self.config.get('log') == 'wandb':
					wandb.log({
						**{f'rl/{key}': value for key, value in rl_stats.items()},
						**{f'sl/{key}': value for key, value in sl_stats.items()},
					})
				else:
					print(f'{rl_stats = }')
					print(f'{sl_stats = }')
		except Exception as e:
			print(e)
		finally:
			if self.config.get('log') == 'wandb':
				wandb.finish()
