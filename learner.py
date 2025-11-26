from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import Model
from multiprocessing import Process
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader

def entropy(logits: torch.Tensor):
	probs = torch.exp(logits)
	logits = torch.where(probs == 0, 0, logits)
	p_log_p = probs * logits
	return -p_log_p.sum(dim=-1)

def KL_divergence(logits: torch.Tensor, old_logits: torch.Tensor):
	probs = torch.exp(logits)
	logits = torch.where(probs == 0, 0, logits - old_logits)
	return (probs * logits).sum(dim=-1)

class SLLearner(Process):
	def __init__(self, model_pool_size, model: Model, device, replay_buffer: ReplayBuffer, config: dict):
		super(SLLearner, self).__init__()
		self.model_pool_size = model_pool_size
		self.model = model
		self.name = model.name
		self.device = device
		self.replay_buffer = replay_buffer
		self.config = config

	def run(self):
		# create model pool
		model_pool = ModelPoolServer(self.model_pool_size, self.name)

		# send to model pool
		self.model.to('cpu')
		model_pool.push(self.model.state_dict()) # push cpu-only tensor to model_pool
		self.model.to(self.device)

		# training
		batch_size = self.config['batch_size']
		mini_batch_size = self.config['mini_batch_size']
		optimizer = torch.optim.Adam(self.model.parameters(), **self.config['optim'])

		cur_time = time.time()
		iterations = 0
		while True:
			# sample batch
			batch = self.replay_buffer.sample(batch_size)
			toks = torch.tensor(batch['toks'], device=self.device)
			output_masks = torch.tensor(batch['output_mask'], device=self.device)
			logits = torch.tensor(batch['logits'], device=self.device)
			targets = F.softmax(logits, dim=-1)
			print(f'{toks.shape=}')
			print(f'{output_masks.shape=}')
			print(f'{targets.shape=}')

			print(f'SL Iteration {iterations}, replay buffer in {self.replay_buffer.stats['sample_in']} out {self.replay_buffer.stats['sample_out']}')
			self.model.train(True) # Training mode

			dataset = TensorDataset(toks, output_masks, targets)
			dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

			for _ in range(self.config['epochs']):
				for tok, output_mask, target in dataloader:
					action_mask = target == -torch.inf
					logit, _ = self.model(tok[..., 0], tok[..., 1])
					logit = torch.where(action_mask, logit, -torch.inf)
					logit = logit.transpose(-1, -2)
					target = target.transpose(-1, -2)
					loss = F.cross_entropy(logit, target, reduction='none')[output_mask].mean()
					optimizer.zero_grad()
					loss.backward()
					if 'clip_grad' in self.config:
						clip_grad_norm_(self.model.parameters(), max_norm=self.config['clip_grad'])
					optimizer.step()

			# push new model
			self.model.to('cpu')
			model_pool.push(self.model.state_dict()) # push cpu-only tensor to model_pool
			self.model.to(self.device)

			# save checkpoints
			t = time.time()
			if t - cur_time > self.config['ckpt_save_interval']:
				path = Path(self.config['ckpt_save_path'], f'{iterations}.pt')
				torch.save(self.model.state_dict(), path)
				cur_time = t
			iterations += 1

class RLLearner(Process):
	def __init__(self, model_pool_size, model: Model, device, replay_buffer: ReplayBuffer, config: dict[str, dict]):
		super(RLLearner, self).__init__()
		self.model_pool_size = model_pool_size
		self.model = model
		self.name = model.name
		self.device = device
		self.replay_buffer = replay_buffer
		self.config = config

	def run(self):
		# create model pool
		model_pool = ModelPoolServer(self.model_pool_size, self.name)

		# send to model pool
		self.model.to('cpu')
		model_pool.push(self.model.state_dict()) # push cpu-only tensor to model_pool
		self.model.to(self.device)

		# training
		eps = self.config['eps']
		value_coef = self.config['value_coef']
		entropy_coef = self.config['entropy_coef']
		batch_size = self.config['batch_size']
		mini_batch_size = self.config['mini_batch_size']
		optimizer = torch.optim.Adam(self.model.parameters(), **self.config['optim'])

		cur_time = time.time()
		iterations = 0
		while True:
			# sample batch
			batch = self.replay_buffer.pop(batch_size)
			toks = torch.tensor(batch['toks'], device=self.device)
			actions = torch.tensor(batch['actions'], device=self.device)
			output_masks = torch.tensor(batch['output_mask'], device=self.device)
			log_probs = torch.tensor(batch['log_probs'], device=self.device)
			logits = torch.tensor(batch['logits'], device=self.device)
			targets = torch.tensor(batch['values'], device=self.device)
			advantages = torch.tensor(batch['advantages'], device=self.device)
			print(f'{toks.shape=}')
			print(f'{actions.shape=}')
			print(f'{output_masks.shape=}')
			print(f'{log_probs.shape=}')
			print(f'{logits.shape=}')
			print(f'{targets.shape=}')
			print(f'{advantages.shape=}')

			print(f'RL Iteration {iterations}, replay buffer in {self.replay_buffer.stats['sample_in']} out {self.replay_buffer.stats['sample_out']}')
			self.model.train(True) # Training mode

			dataset = TensorDataset(toks, actions, output_masks, log_probs, logits, targets, advantages)
			dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

			for _ in range(self.config['epochs']):
				for tok, action, output_mask, old_log_prob, old_logit, target, adv in dataloader:
					action_mask = target == -torch.inf
					logit, value = self.model(tok[..., 0], tok[..., 1])
					logit = torch.where(action_mask, logit, -torch.inf)
					logit = F.log_softmax(logit, dim=-1)
					log_prob = logit.gather(-1, action.unsqueeze(dim=-1)).squeeze(dim=-1)
					ratio = torch.exp(log_prob - old_log_prob)
					surr1 = ratio * adv
					surr2 = torch.clip(ratio, 1 - eps, 1 + eps) * adv
					policy_loss = -torch.min(surr1, surr2)[output_mask].mean()
					value_loss = F.mse_loss(value, target, reduction='none')[output_mask].mean()
					entropy_loss = -entropy(logits)[output_mask].mean()
					KL_div = KL_divergence(logit, old_logit)[output_mask].mean()
					loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
					optimizer.zero_grad()
					loss.backward()
					if 'clip_grad' in self.config:
						clip_grad_norm_(self.model.parameters(), max_norm=self.config['clip_grad'])
					optimizer.step()

			# push new model
			self.model.to('cpu')
			model_pool.push(self.model.state_dict()) # push cpu-only tensor to model_pool
			self.model.to(self.device)

			# save checkpoints
			t = time.time()
			if t - cur_time > self.config['ckpt_save_interval']:
				path = Path(self.config['ckpt_save_path'], f'{iterations}.pt')
				torch.save(self.model.state_dict(), path)
				cur_time = t
			iterations += 1
