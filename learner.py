from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import Model
from multiprocessing import Process
from pathlib import Path
import threading
import time
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

def masked_normalize(advantages: torch.Tensor, output_mask: torch.Tensor, eps = 1e-5):
	mean = advantages[output_mask].mean()
	std = advantages[output_mask].std() + eps
	return (advantages - mean) / std

class SLLearner:
	def __init__(self, name: str, models: dict[str, Model], replay_buffers: dict[str, ReplayBuffer], device, config: dict):
		self.name = name
		self.model = models[name].to(device)
		self.replay_buffer = replay_buffers[name]
		self.device = device
		self.config = config
		self.model_pool = ModelPoolServer(config['model_pool_size'], name)
		self.model_pool.push(self.model.state_dict())
		self.optimizer = torch.optim.Adam(self.model.parameters(), **config['optim'])
		self.iterations = 0
	
	def step(self):
		batch_size = self.config['batch_size']
		mini_batch_size = self.config['mini_batch_size']

		# sample batch
		batch = self.replay_buffer.sample(batch_size)
		toks = torch.tensor(batch['toks'], device=self.device)
		output_masks = torch.tensor(batch['output_mask'], device=self.device)
		logits = torch.tensor(batch['logits'], device=self.device)
		print(f'{toks.shape=}')
		print(f'{output_masks.shape=}')
		print(f'{logits.shape=}')

		print(f'SL Iteration {self.iterations}, replay buffer in {self.replay_buffer.stats["sample_in"]} out {self.replay_buffer.stats["sample_out"]}')
		self.model.train(True) # Training mode

		dataset = TensorDataset(toks, output_masks, logits)
		dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

		for _ in range(self.config['epochs']):
			for tok, output_mask, old_logit in dataloader:
				action_mask = old_logit == -torch.inf
				target = F.softmax(old_logit, dim=-1)
				logit, _ = self.model(tok[..., 0], tok[..., 1])
				logit = torch.where(action_mask, logit, -torch.inf)
				# Cross entropy expects (B, C, Seq)
				logit = logit.transpose(-1, -2)
				target = target.transpose(-1, -2)
				loss = F.cross_entropy(logit, target, reduction='none')[output_mask].mean()

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

class RLLearner:
	def __init__(self, name: str, models: dict[str, Model], replay_buffers: dict[str, ReplayBuffer], device, config: dict):
		self.name = name
		self.model = models[name].to(device)
		self.replay_buffer = replay_buffers[name]
		self.device = device
		self.config = config
		self.model_pool = ModelPoolServer(config['model_pool_size'], name)
		self.model_pool.push(self.model.state_dict())
		self.optimizer = torch.optim.Adam(self.model.parameters(), **config['optim'])
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
		print(f'{toks.shape=}')
		print(f'{actions.shape=}')
		print(f'{output_masks.shape=}')
		print(f'{log_probs.shape=}')
		print(f'{logits.shape=}')
		print(f'{targets.shape=}')
		print(f'{advantages.shape=}')

		print(f'RL Iteration {self.iterations}, replay buffer in {self.replay_buffer.stats["sample_in"]} out {self.replay_buffer.stats["sample_out"]}')
		self.model.train(True) # Training mode

		dataset = TensorDataset(toks, actions, output_masks, log_probs, logits, targets, advantages)
		dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

		for _ in range(self.config['epochs']):
			for tok, action, output_mask, old_log_prob, old_logit, target, adv in dataloader:
				action_mask = old_logit == -torch.inf
				logit, value = self.model(tok[..., 0], tok[..., 1])
				logit = torch.where(action_mask, logit, -torch.inf)
				logit = F.log_softmax(logit, dim=-1)
				log_prob = logit.gather(-1, action.unsqueeze(dim=-1)).squeeze(dim=-1)
				ratio = torch.exp(log_prob - old_log_prob)
				surr1 = ratio * adv
				surr2 = torch.clip(ratio, 1 - eps, 1 + eps) * adv
				policy_loss = -torch.min(surr1, surr2)[output_mask].mean()
				value_loss = F.mse_loss(value, target, reduction='none')[output_mask].mean()
				entropy_loss = -entropy(logit)[output_mask].mean()
				KL_div = KL_divergence(logit, old_logit)[output_mask].mean()
				loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

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

class Learner(Process):
	def __init__(self, models: dict[str, Model], replay_buffers: dict[str, ReplayBuffer], device, config: dict):
		super(Learner, self).__init__()
		self.sl_learner = SLLearner('avg', models, replay_buffers, device, config['sl'])
		self.rl_learner = RLLearner('best', models, replay_buffers, device, config['rl'])

	def _flush(self):
		while True:
			self.sl_learner.replay_buffer._flush()
			self.rl_learner.replay_buffer._flush()
			time.sleep(0.1)

	def run(self):
		thread = threading.Thread(target=self._flush, daemon=True)
		thread.start()
		while True:
			self.rl_learner.step()
			self.sl_learner.step()
