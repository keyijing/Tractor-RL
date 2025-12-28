from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import Model
from multiprocessing import Process
from pathlib import Path
import wandb
import traceback
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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

class RLLearner:
	def __init__(self, model: Model, device_id: int, replay_buffer: ReplayBuffer, config: dict):
		self.config = config
		self.master = dist.get_rank() == 0
		self.world_size = dist.get_world_size()
		self.device = f'cuda:{device_id}'
		model = model.to(self.device)
		if 'load_path' in config:
			print(f'load model from {config["load_path"]}')
			model.load_state_dict(torch.load(config['load_path'], map_location=self.device))
		self.model = DDP(model, device_ids=[device_id])
		self.replay_buffer = replay_buffer

		if self.master:
			print('creating model pool server')
			self.model_pools = {
				'best': ModelPoolServer(config['model_pool']['best']['size'], 'best'),
				'ckpt': ModelPoolServer(config['model_pool']['ckpt']['size'], 'ckpt'),
			}
			print('server create done')
			self.model_pools['best'].push(self.model.module.state_dict())
			self.model_pools['ckpt'].push(self.model.module.state_dict())

		self.optimizer = torch.optim.AdamW(self.model.parameters(), **config['optim'])
		self.iterations = 0

	def step(self):
		eps = self.config['eps']
		value_coef = self.config['value_coef']
		entropy_coef = self.config['entropy_coef']
		aux_coef = self.config['aux_coef']
		batch_size = self.config['batch_size']
		mini_batch_size = self.config['mini_batch_size']

		# sample batch
		local_batch_size = batch_size // self.world_size
		if self.master:
			batch = self.replay_buffer.pop(batch_size)
			reward = np.mean(np.where(batch['output_mask'], batch['rewards'], 0).sum(axis=-1)).item()
			del batch['rewards']
			for key in batch:
				batch[key] = torch.tensor(batch[key], device=self.device)
			batch['advantages'] = masked_normalize(batch['advantages'], batch['output_mask'])
			for key in batch:
				batch[key] = list(batch[key].chunk(self.world_size))
		else:
			batch = defaultdict(lambda: None)

		raw_model = self.model.module
		max_seq_len = raw_model.max_seq_len
		n_actions = raw_model.n_actions

		toks = torch.empty((local_batch_size, max_seq_len, 2), dtype=torch.int64, device=self.device)
		actions = torch.empty((local_batch_size, max_seq_len), dtype=torch.int64, device=self.device)
		output_masks = torch.empty((local_batch_size, max_seq_len), dtype=bool, device=self.device)
		log_probs = torch.empty((local_batch_size, max_seq_len), dtype=torch.float32, device=self.device)
		# Assume logits are actually log_probs (log_softmax output)
		logits = torch.empty((local_batch_size, max_seq_len, n_actions), dtype=torch.float32, device=self.device)
		targets = torch.empty((local_batch_size, max_seq_len), dtype=torch.float32, device=self.device)
		advantages = torch.empty((local_batch_size, max_seq_len), dtype=torch.float32, device=self.device)

		dist.scatter(toks, batch['toks'])
		dist.scatter(actions, batch['actions'])
		dist.scatter(output_masks, batch['output_mask'])
		dist.scatter(log_probs, batch['log_probs'])
		dist.scatter(logits, batch['logits'])
		dist.scatter(targets, batch['values'])
		dist.scatter(advantages, batch['advantages'])

		if self.master:
			print(f'RL Iteration {self.iterations}, replay buffer in {self.replay_buffer.stats["sample_in"]} out {self.replay_buffer.stats["sample_out"]}')
		self.model.train(True) # Training mode

		dataset = TensorDataset(toks, actions, output_masks, log_probs, logits, targets, advantages)
		dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

		stats = defaultdict(list)
		if self.master:
			stats['reward'] = reward
		else:
			stats['reward'] = 0

		for _ in range(self.config['epochs']):
			for tok, action, output_mask, old_log_prob, old_logit, target, adv in dataloader:
				total = output_mask.sum()
				def _masked_mean(x: torch.Tensor):
					return torch.where(output_mask, x, 0).sum() / total

				action_mask = old_logit != -torch.inf
				logit, value, mask_pred = self.model(tok[..., 0], tok[..., 1])
				logit = torch.where(action_mask, logit, -torch.inf)
				logit = F.log_softmax(logit, dim=-1)
				log_prob = logit.gather(-1, action.unsqueeze(dim=-1)).squeeze(dim=-1)
				ratio = torch.exp(log_prob - old_log_prob)
				surr1 = ratio * adv
				surr2 = torch.clip(ratio, 1 - eps, 1 + eps) * adv

				pos_weight = torch.sum(action_mask, dim=-1, keepdim=True)
				pos_weight = torch.clip((n_actions - pos_weight) / pos_weight, max=10)

				policy_loss = -_masked_mean(torch.min(surr1, surr2))
				value_loss = _masked_mean(F.mse_loss(value, target, reduction='none'))
				entropy_loss = -_masked_mean(entropy(logit))
				aux_loss = _masked_mean(F.binary_cross_entropy_with_logits(
					mask_pred, action_mask.to(dtype=torch.float32), reduction='none', pos_weight=pos_weight
				).mean(dim=-1))
				KL_div = _masked_mean(KL_divergence(logit, old_logit))
				loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + aux_coef * aux_loss

				stats['policy_loss'].append(policy_loss.item())
				stats['value_loss'].append(value_loss.item())
				stats['entropy_loss'].append(entropy_loss.item())
				stats['aux_loss'].append(aux_loss.item())
				stats['total_loss'].append(loss.item())
				stats['kl_div'].append(KL_div.item())

				self.optimizer.zero_grad()
				loss.backward()
				if 'clip_grad' in self.config:
					clip_grad_norm_(self.model.parameters(), max_norm=self.config['clip_grad'])
				self.optimizer.step()

		# push new model
		if self.master:
			self.model_pools['best'].push(self.model.module.state_dict())
			self.iterations += 1
			if self.iterations % self.config['ckpt_save_interval'] == 0:
				self.model_pools['ckpt'].push(self.model.module.state_dict())
				path = Path(self.config['ckpt_save_path'], f'{self.iterations}.pt')
				torch.save(self.model.module.state_dict(), path)

		for key, value in stats.items():
			if key == 'reward':
				continue
			value = torch.tensor(sum(value) / len(value), dtype=torch.float32, device=self.device)
			dist.reduce(value, dst=0, op=dist.ReduceOp.AVG)
			stats[key] = value.item()
		return stats

class Learner(Process):
	def __init__(self, rank: int, selected_gpus: list[int], replay_buffer: ReplayBuffer, config: dict):
		super(Learner, self).__init__()
		self.rank = rank
		self.world_size = len(selected_gpus)
		self.device_id = selected_gpus[rank]
		self.replay_buffer = replay_buffer
		self.config = config

	def run(self):
		master = self.rank == 0
		if master and self.config.get('log') == 'wandb':
			time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
			wandb.init(project='Tractor-RL', name=f'Learner-{time}', config=self.config)

		torch.cuda.set_device(self.device_id)
		dist.init_process_group(
			backend='nccl',
			init_method='tcp://127.0.0.1:29500',
			world_size=self.world_size,
			rank=self.rank,
		)

		try:
			model = Model(**self.config['model'])
			self.rl_learner = RLLearner(model, self.device_id, self.replay_buffer, self.config['rl'])

			while True:
				rl_stats = self.rl_learner.step()
				if master:
					if self.config.get('log') == 'wandb':
						wandb.log({
							**{f'rl/{key}': value for key, value in rl_stats.items()}
						})
					else:
						print(f'{rl_stats = }')
		except Exception as _:
			traceback.print_exc()
		finally:
			dist.destroy_process_group()
			if master and self.config.get('log') == 'wandb':
				wandb.finish()
