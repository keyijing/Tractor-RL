from model import TensorBuffer, collate_kv_cache, Model
from agent import N_TOKENS, N_ACTIONS
from env import Env
from collections import defaultdict
import numpy as np
import torch

def eval_models(
	models: dict[str, Model],
	model_ids: list[str],
	envs: list[Env],
	tensor_buffer: TensorBuffer,
	device,
):
	batch_size = len(envs)
	kv_caches = [
		[models[mid].transformer.create_kv_cache(tensor_buffer) for mid in model_ids]
		for _ in range(batch_size)
	]
	for env in envs:
		env.reset()
	done_cnt = 0

	while done_cnt < batch_size:

		# Grouping buckets: model_index -> list of data
		# We store the environment object and trajectory directly in the bucket
		batches: dict[str] = defaultdict(lambda: {
			'toks': [], 'masks': [], 'lens': [],
			'envs': [], 'caches': [],
		})

		# --- A. Gather Observations (Bucket by Model) ---
		# Iterate all environments. If done, skip.
		for env_caches, env in zip(kv_caches, envs):
			if env.done:
				continue
			obs = env.obs()
			pid: int = obs['player']
			mid = model_ids[pid] # Get model index for this player

			# Add to specific model bucket
			group = batches[mid]
			group['toks'].append(obs['toks'])
			group['masks'].append(obs['action_mask'])
			group['lens'].append(len(obs['toks']))
			group['envs'].append(env)
			group['caches'].append(env_caches[pid])

		# --- B. Process Each Model Group ---
		for mid, group in batches.items():
			toks = group['toks']
			action_masks = group['masks']
			valid_lengths = group['lens']
			curr_envs = group['envs']
			caches = group['caches']

			# --- I. Prepare Tensors ---
			# Pad sequences (Ragged Batching)
			# toks_tensor: (Current_Batch, Max_Len_In_Batch, 2)
			toks_tensor = torch.nn.utils.rnn.pad_sequence(
				[torch.tensor(tok, dtype=torch.int64) for tok in toks],
				batch_first=True
			).to(device)

			action_mask_tensor = torch.from_numpy(np.stack(action_masks)).to(device)

			# Collate KV cache for this specific subset of players
			kv_cache = collate_kv_cache(caches)

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

			# --- VI. Step Environment & Store ---
			for env, action in zip(curr_envs, actions):
				done = env.step(action)
				if done:
					done_cnt += 1

	# Final Rewards
	scores_tot = [0 for _ in range(4)]
	for env in envs:
		scores = env.game.final_scores
		for i in range(4):
			scores_tot[i] += scores[i]
	return [score / batch_size for score in scores_tot]

if __name__ == '__main__':
	seed = 10
	batch_size = 128
	device = 'cuda:5'

	# Load models
	conf = {
		'n_toks': N_TOKENS,
		'n_players': 4,
		'n_actions': N_ACTIONS,
		'd_model': 256,
		'max_seq_len': 384,
		'num_blocks': 16,
		'num_heads': 8,
	}
	models = {
		'1': Model(**conf).to(device),
		'0': Model(**conf).to(device),
	}
	for model in models.values():
		model.eval()
	models['1'].load_state_dict(torch.load('checkpoint/9900.pt', map_location=device))
	models['0'].load_state_dict(torch.load('checkpoint/6500.pt', map_location=device))

	# Setup Single Shared TensorBuffer
	# Capacity: batch_size * 4 players * num_blocks * 2 (k+v)
	# We use ref_model for shape/config since all models are the same type.
	ref_model = next(iter(models.values()))
	capacity = batch_size * 4 * ref_model.transformer.num_blocks * 2
	tensor_buffer = TensorBuffer(capacity, ref_model.transformer.cache_shape, device=device)

	# Create envs
	envs = [Env(seed + i) for i in range(batch_size)]

	scores_tot = [0 for _ in range(4)]
	for _ in range(16):
		print(_)
		scores = eval_models(models, ['0', '1', '0', '1'], envs, tensor_buffer, device)
		for i in range(4):
			scores_tot[i] += scores[i] / 16
	print(scores_tot)
