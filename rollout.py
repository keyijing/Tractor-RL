from model import TensorBuffer, collate_kv_cache, Model
from agent import N_TOKENS, N_ACTIONS
from env import Env
import numpy as np
import torch
from collections import defaultdict

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

def rollout(models: list[Model], model_ids: list[int], batch_size = 1, device = None):
	"""
	models: List of Model objects (must share same architecture/hyperparams).
	model_ids: List[int] of length 4. Maps player_id -> model_index.
	"""
	# 1. Setup Single Shared TensorBuffer
	# Capacity: batch_size * 4 players * num_blocks * 2 (k+v)
	# We use models[0] for shape/config since all models are the same type.
	ref_model = models[0]
	capacity = batch_size * 4 * ref_model.transformer.num_blocks * 2
	tensor_buffer = TensorBuffer(capacity, ref_model.transformer.cache_shape, device=device)

	# 2. Setup Shared CPU Buffers
	# Capacity: 4 players per env * batch_size
	np_buffers = {
		name: NumpyBuffer(batch_size * 4, ref_model.max_seq_len, dtype=dtype)
		for name, dtype in {
			'actions': np.int64,
			'output_mask': bool,
			'log_probs': np.float32,
			'values': np.float32,
			'rewards': np.float32,
			'advantages': np.float32,
		}.items()
	}
	np_buffers['toks'] = NumpyBuffer(batch_size * 4, (ref_model.max_seq_len, 2), dtype=np.int64)
	np_buffers['logits'] = NumpyBuffer(batch_size * 4, (ref_model.max_seq_len, ref_model.n_actions), dtype=np.float32)

	# 3. Create Trajectories
	# traj_all[env_index][player_index]
	traj_all = [[Trajectory(ref_model, tensor_buffer, np_buffers) for _ in range(4)] for _ in range(batch_size)]
	envs = [Env() for _ in range(batch_size)]

	# 4. Infinite Loop
	while True:

		timer = CumulativeTimer()
		timer.__enter__()

		# Reset Cache and Buffers
		for i in range(batch_size):
			for j in range(4):
				for cache in traj_all[i][j].kv_cache:
					cache.reset()
		np_buffers['output_mask'].zero_()
		for env in envs:
			env.reset()
		done_cnt = 0

		while done_cnt < batch_size:

			# Grouping buckets: model_index -> list of data
			# We store the environment object and trajectory directly in the bucket
			batches: dict[int] = defaultdict(lambda: {
				'toks': [], 'masks': [], 'lens': [], 
				'envs': [], 'trajs': []
			})

			# --- A. Gather Observations (Bucket by Model) ---
			# Iterate all environments. If done, skip.
			for env_trajs, env in zip(traj_all, envs):
				if env.done:
					continue
				obs = env.obs()
				pid: int = obs['player']
				mid = model_ids[pid] # Get model index for this player

				traj = env_trajs[pid]

				# Store reward from previous action
				traj.set_reward(obs['reward'])

				# Add to specific model bucket
				group = batches[mid]
				group['toks'].append(obs['toks'])
				group['masks'].append(obs['action_mask'])
				group['lens'].append(len(obs['toks']))
				group['envs'].append(env)
				group['trajs'].append(traj)

			# --- B. Process Each Model Group ---
			for mid, group in batches.items():
				toks = group['toks']
				action_masks = group['masks']
				valid_lengths = group['lens']
				curr_envs = group['envs']
				trajs = group['trajs']

				# --- I. Prepare Tensors ---
				# Pad sequences (Ragged Batching)
				# toks_tensor: (Current_Batch, Max_Len_In_Batch, 2)
				toks_tensor = torch.nn.utils.rnn.pad_sequence(
					[torch.tensor(tok, dtype=torch.int64) for tok in toks],
					batch_first=True
				).to(device)

				action_mask_tensor = torch.from_numpy(np.stack(action_masks)).to(device)

				# Collate KV cache for this specific subset of players
				kv_cache = collate_kv_cache([traj.kv_cache for traj in trajs])

				# --- II. Inference (Using the specific model weights) ---
				# toks_tensor[..., 0] is Token ID, [..., 1] is Player ID
				ret = models[mid].get_action_and_value(
					toks_tensor[..., 0],
					toks_tensor[..., 1],
					action_mask_tensor,
					kv_cache,
					valid_lengths
				)

				# --- III. Move to CPU ---
				actions = ret['actions'].tolist()
				log_probs = ret['log_probs'].tolist()
				values = ret['values'].tolist()
				# Copy logits carefully; if large vocab, this is slow.
				# Since this is RL, N_ACTIONS usually small.
				logits = ret['logits'].cpu().numpy()

				# --- VI. Step Environment & Store ---
				for env, traj, seq_len, tok, action, logit, log_prob, value in zip(
					curr_envs, trajs, valid_lengths, toks, actions, logits, log_probs, values
				):
					done = env.step(action)
					if done:
						done_cnt += 1

					# Store trajectory data
					traj.append(seq_len, tok, action, log_prob, logit, value)

		# 5. Final Rewards
		# Since the loop breaks when env.step returns Done, the last reward 
		# hasn't been observed via env.obs().
		for env, trajs in zip(envs, traj_all):
			for reward, traj in zip(env.rewards, trajs):
				traj.set_reward(reward)

		# 6. GAE Calculation
		compute_gae(np_buffers, 0.99, 0.95)

		timer.__exit__(None, None, None)
		print(timer.get_total_time())

		# TODO: Add the trajectories to dataset

if __name__ == '__main__':
	device = 'cuda'
	model = Model(N_TOKENS, 4, N_ACTIONS, d_model=256, max_seq_len=384, num_blocks=8, num_heads=8).to(device)
	model2 = Model(N_TOKENS, 4, N_ACTIONS, d_model=256, max_seq_len=384, num_blocks=8, num_heads=8).to(device)
	model.eval()
	rollout([model, model2], [0, 1, 0, 1], 64, device)
