from model import TensorBuffer, collate_kv_cache, Model
from agent import N_TOKENS, N_ACTIONS
from env import Env
import numpy as np
import torch

class NumpyBuffer:
	"""
	A simple memory pool for CPU-side numpy arrays to reduce allocation overhead.
	"""
	def __init__(self, capacity: int, shape: tuple | int, dtype=np.float32):
		if isinstance(shape, int):
			shape = (shape,)
		self.shape = shape
		self.capacity = capacity
		self.buffer = np.empty(shape=(capacity,) + shape, dtype=dtype)
		self.available = list(reversed(range(capacity)))

	def zero_(self):
		self.buffer.fill(0)

	def allocate(self):
		if not self.available:
			raise MemoryError('bad alloc: no available rows')
		return self.available.pop()

	def deallocate(self, idx):
		self.available.append(idx)

	def __getitem__(self, idx):
		return self.buffer[idx]

	def __setitem__(self, idx, value):
		self.buffer[idx] = value

class Trajectory:
	"""
	Manages the data storage for a single player in a single environment.
	"""
	def __init__(self, model: Model, tensor_buffer: TensorBuffer, np_buffers: dict[str, NumpyBuffer]):
		# Allocate GPU KV-Cache rows
		self.kv_cache = model.transformer.create_kv_cache(tensor_buffer)

		# Helper to allocate a row from the global NumpyBuffers
		def _allocate(name: str) -> np.ndarray:
			return np_buffers[name].buffer[np_buffers[name].allocate()]

		# Allocate pointers to CPU buffers
		self.toks = _allocate('toks')
		self.actions = _allocate('actions')
		self.output_mask = _allocate('output_mask')
		self.log_probs = _allocate('log_probs')
		self.logits = _allocate('logits')
		self.values = _allocate('values')
		self.rewards = _allocate('rewards')

		self.max_len = self.toks.shape[0]
	
	def set_reward(self, last_reward: float):
		"""
		Assigns the reward received in the CURRENT observation to the PREVIOUS step's action.
		"""
		# kv_cache.curr_pos has NOT been updated for the current step yet.
		# So curr_pos points to the end of the previous sequence.
		# pos = curr_pos - 1 is the index of the last token generated in the previous step.
		pos = self.kv_cache[0].curr_pos - 1
		if pos >= 0:
			assert self.output_mask[pos] == True
			self.rewards[pos] = last_reward
	
	def append(self, seq_len: int, toks: list, action: int, log_prob: float, logits: np.ndarray, value: float):
		"""
		Records the data for the current step AFTER model inference.
		"""
		# kv_cache.curr_pos HAS been updated by the model forward pass.
		# It now includes the new tokens.
		curr_end = self.kv_cache[0].curr_pos
		pos = curr_end - 1 # The index where the action was taken (last token)

		if curr_end > self.max_len:
			raise RuntimeError(f"Episode length {curr_end} exceeds buffer size {self.max_len}")

		# Store the token sequence (filling backwards from pos)
		self.toks[pos - seq_len + 1 : pos + 1] = toks

		# Store metadata at the exact action position
		self.actions[pos] = action
		self.output_mask[pos] = True
		self.log_probs[pos] = log_prob
		self.logits[pos] = logits
		self.values[pos] = value

from timer import CumulativeTimer

def compute_gae(buffers: dict[str, NumpyBuffer], gamma, lam):
	advantages = buffers['advantages']
	advantages.zero_()

	rewards = buffers['rewards']
	values = buffers['values']
	output_mask = buffers['output_mask']

	batch_size = advantages.capacity
	seq_len = advantages.shape[0]

	last_gae = np.zeros(batch_size, dtype=np.float32)
	last_value = np.zeros(batch_size, dtype=np.float32)

	# Iterate backwards through time
	for i in reversed(range(seq_len)):
		# GAE Formula: delta = r_t + gamma * V_{t+1} - V_t
		delta = rewards[:, i] + gamma * last_value - values[:, i]

		# If output_mask is False (padding/intermediate token), pass through gradients/values 
		# but don't compute new advantage.
		# Actually, for PPO with masked actions, we usually force advantage to 0 at masked steps,
		# but passing 'last_gae' through allows gradients to flow if chunks were connected (not case here).
		# Standard Masking:
		gae = np.where(output_mask[:, i], delta + gamma * lam * last_gae, last_gae)
		advantages[:, i] = gae
		last_gae = gae
		# Update last_value: if this step was real, it becomes V_{t+1} for the next step (i-1)
		last_value = np.where(output_mask[:, i], values[:, i], last_value)

	# Compute Returns: Return = Advantage + Value
	values[:] += advantages[:]

def rollout(model: Model, batch_size = 1, device = None):
	# 1. Setup Memory
	# Capacity: 4 players per env * batch_size * blocks * (k_cache + v_cache)
	tensor_buffer = TensorBuffer(batch_size * model.transformer.num_blocks * 8, model.transformer.cache_shape, device=device)

	# CPU Buffers
	# Capacity: 4 players per env * batch_size
	np_buffers = {
		name: NumpyBuffer(batch_size * 4, model.max_seq_len, dtype=dtype)
		for name, dtype in {
			'actions': np.int64,
			'output_mask': bool,
			'log_probs': np.float32,
			'values': np.float32,
			'rewards': np.float32,
			'advantages': np.float32,
		}.items()
	}
	np_buffers['toks'] = NumpyBuffer(batch_size * 4, (model.max_seq_len, 2), dtype=np.int64)
	np_buffers['logits'] = NumpyBuffer(batch_size * 4, (model.max_seq_len, model.n_actions), dtype=np.float32)

	# Create Objects
	# traj_all[env_index][player_index]
	traj_all = [[Trajectory(model, tensor_buffer, np_buffers) for _ in range(4)] for _ in range(batch_size)]
	envs = [Env() for _ in range(batch_size)]

	# 2. Infinite Training Loop
	while True:
		timer1 = CumulativeTimer()
		timer2 = CumulativeTimer()
		timer3 = CumulativeTimer()
		timer4 = CumulativeTimer()
		timer5 = CumulativeTimer()

		# Reset Logic
		for i in range(batch_size):
			for j in range(4):
				for cache in traj_all[i][j].kv_cache:
					cache.reset()
		# Zero out masks (important because we reuse the buffer)
		np_buffers['output_mask'].zero_()
		for env in envs:
			env.reset()
		done_cnt = 0
		steps = 0

		# 3. Episode Loop
		while done_cnt < batch_size:
			steps += 1
			trajs: list[Trajectory] = []
			curr_envs: list[Env] = []
			toks = []
			action_masks = []

			# --- A. Gather Observations ---
			with timer1:
				# Iterate all environments. If done, skip.
				# This naturally reduces the batch size sent to the GPU as envs finish.
				for traj_one, env in zip(traj_all, envs):
					if env.done:
						continue
					obs = env.obs()
					player: int = obs['player']
					last_reward = obs['reward']
					tok = obs['toks'] # Shape: (Len, 2)
					action_mask = obs['action_mask']

					# Identify correct trajectory for this (env, player)
					traj = traj_one[player]
					# 1. Assign Reward to previous step
					traj.set_reward(last_reward)

					trajs.append(traj)
					curr_envs.append(env)
					toks.append(tok)
					action_masks.append(action_mask)

			# --- B. Prepare Tensors ---
			with timer2:
				# Pad sequences (Ragged Batching)
				# toks_tensor: (Current_Batch, Max_Len_In_Batch, 2)
				toks_tensor = torch.nn.utils.rnn.pad_sequence(
					[torch.tensor(tok, dtype=torch.int64) for tok in toks],
					batch_first=True
				).to(device)

				action_mask_tensor = torch.from_numpy(np.stack(action_masks)).to(device)

				# KV Cache: We pass the list of caches corresponding strictly 
				# to the currently active environments/players.
				kv_cache = collate_kv_cache([traj.kv_cache for traj in trajs])
				valid_lengths = [len(tok) for tok in toks]

			# --- C. Model Inference ---
			with timer3:
				# toks_tensor[..., 0] is Token ID, [..., 1] is Player ID
				ret = model.get_action_and_value(
					toks_tensor[..., 0],
					toks_tensor[..., 1],
					action_mask_tensor,
					kv_cache,
					valid_lengths
				)

			# --- D. Move to CPU ---
			with timer4:
				actions = ret['actions'].tolist()
				logits = ret['logits'].cpu().numpy()
				log_probs = ret['log_probs'].tolist()
				# Copy logits carefully; if large vocab, this is slow.
				# Since this is RL, N_ACTIONS usually small.
				values = ret['values'].tolist()

			# --- E. Step & Store ---
			with timer5:
				for env, traj, seq_len, tok, action, logit, log_prob, value in zip(
					curr_envs, trajs, valid_lengths, toks, actions, logits, log_probs, values
				):
					done = env.step(action)
					if done:
						done_cnt += 1

					# 2. Append new data
					traj.append(seq_len, tok, action, log_prob, logit, value)

		# 4. Final Rewards
		# Since the loop breaks when env.step returns Done, the last reward 
		# hasn't been observed via env.obs().
		for env, trajs in zip(envs, traj_all):
			for reward, traj in zip(env.rewards, trajs):
				traj.set_reward(reward)

		# 5. GAE Calculation
		timer6 = CumulativeTimer()
		with timer6:
			compute_gae(np_buffers, 0.99, 0.95)
		
		print(timer1.get_total_time(), timer2.get_total_time(), timer3.get_total_time(), timer4.get_total_time(), timer5.get_total_time(), timer6.get_total_time(), sep='\n')
		print(steps)
		# TODO: Add the trajectories to dataset

if __name__ == '__main__':
	device = 'cuda'
	model = Model(N_TOKENS, 4, N_ACTIONS, d_model=256, max_seq_len=384, num_blocks=8, num_heads=8).to(device)
	model.eval()
	rollout(model, 64, device)
