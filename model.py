from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(module: nn.Module):
	if isinstance(module, nn.Linear):
		nn.init.normal_(module.weight, std=module.in_features ** -0.5)
		if module.bias is not None:
			nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Embedding):
		nn.init.normal_(module.weight, std=module.embedding_dim ** -0.5)
		if module.padding_idx is not None:
			with torch.no_grad():
				module.weight[module.padding_idx].zero_()

class TensorBuffer:
	"""
	A pre-allocated block of GPU memory acting as a memory pool.
	"""
	def __init__(self, capacity: int, size: torch.Size, dtype=torch.float, device=None):
		self.size = size
		self.capacity = capacity
		# Shape: (Capacity, Heads, MaxSeqLen, Dim)
		self.buffer = torch.empty((capacity,) + size, dtype=dtype, device=device)
		# Simple stack for memory management
		self.available = list(reversed(range(capacity)))

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

class KVCacheRow:
	"""
	Logical handle for a single sequence's KV history.
	Holds pointers to specific rows in the TensorBuffer.
	"""
	def __init__(self, buffer: TensorBuffer):
		self.buffer = buffer
		self.k_index = buffer.allocate()
		self.v_index = buffer.allocate()
		self.curr_pos = 0

	def reset(self):
		self.curr_pos = 0

	def __del__(self):
		# Return slots to pool when the object is garbage collected
		self.buffer.deallocate(self.v_index)
		self.buffer.deallocate(self.k_index)

class KVCache:
	"""
	Batch operator for KVCacheRows. 
	Manages vectorized updates and retrievals for a specific layer.
	"""
	def __init__(self, caches: List[KVCacheRow]):
		self.caches = caches
		self.buffer = caches[0].buffer
		self.batch_size = len(caches)

		# Store indices as tensors for Advanced Indexing
		self.device = self.buffer.buffer.device
		self.k_indices = torch.tensor([c.k_index for c in caches], dtype=torch.int64, device=self.device)
		self.v_indices = torch.tensor([c.v_index for c in caches], dtype=torch.int64, device=self.device)

	def update(self, k_new: torch.Tensor, v_new: torch.Tensor, pos_ids: torch.Tensor, valid_lengths: List[int]):
		"""
		k_new, v_new: (Batch, Heads, Seq_Len, Dim)
		pos_ids: (Batch, Seq_Len)- Target write positions
		valid_lengths: (Batch,) - Actual length to increment
		"""
		# 1. Update GPU Buffer (Advanced Indexing)
		# Advanced Indexing Note:
		# self.buffer shape: (Capacity, Heads, MaxLen, Dim)
		# Indexing: buffer[Indices(B,1), :, Pos(B,S), :]
		# PyTorch moves indexed dimensions to the front: Result shape is (Batch, Seq_Len, Heads, Dim)
		# Therefore, we must transpose k_new from (B, H, S, D) to (B, S, H, D) to match.
		self.buffer[self.k_indices.unsqueeze(1), :, pos_ids, :] = k_new.transpose(1, 2)
		self.buffer[self.v_indices.unsqueeze(1), :, pos_ids, :] = v_new.transpose(1, 2)

		# We must update the underlying KVCacheRow objects so they stay valid
		for cache, length in zip(self.caches, valid_lengths):
			cache.curr_pos += length

	def fetch(self):
		"""
		Returns a view up to the deepest valid position in the batch.
		"""
		max_pos = max(cache.curr_pos for cache in self.caches)
		return (
			self.buffer[self.k_indices, :, :max_pos, :],
			self.buffer[self.v_indices, :, :max_pos, :]
		)

	def get_offsets(self):
		return torch.tensor([cache.curr_pos for cache in self.caches], dtype=torch.int64, device=self.device)

def collate_kv_cache(kv_cache_rows_list: List[List[KVCacheRow]]):
	"""
	Converts a list of [Block0_Row, Block1_Row...] into a list of [Block0_BatchCache, Block1_BatchCache...]
	"""
	num_blocks = len(kv_cache_rows_list[0])
	# Transpose list of lists
	return [KVCache([cache[i] for cache in kv_cache_rows_list]) for i in range(num_blocks)]

class RoPE(nn.Module):
	def __init__(self, dim, max_seq_len, base = 10000):
		super().__init__()
		self.dim = dim

		# Create position index [0, 1, 2, ..., maxlen-1]
		position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(dim=-1)

		# Calculate frequency factor
		dim_index = torch.arange(0, dim, 2, dtype=torch.float)
		dim_index = torch.pow(base, -dim_index / dim)

		# Calculate product of position and frequency factors
		freqs = position * dim_index

		# Create complex rotation factors (MaxSeqLen, Dim/2)
		self.freqs_cis = nn.Parameter(torch.polar(torch.ones_like(freqs), freqs), requires_grad=False)

	def forward(self, x, pos_ids: torch.Tensor = None):
		"""
		x: (Batch, Heads, Seq_Len, Head_Dim)
		pos_ids: (Batch, Seq_Len)
		"""
		batch_size, num_heads, seq_len, head_dim = x.shape

		# Separate real and imaginary parts
		x_complex = torch.view_as_complex(
			x.reshape(batch_size, num_heads, seq_len, head_dim//2, 2)
		)

		if pos_ids is None:
			# Standard sequential (1, Seq_Len, Dim/2)
			current_freqs = self.freqs_cis[:seq_len].unsqueeze(0)
		else:
			# Select the specific rotation frequencies for the current positions
			# pos_ids: (Batch, Seq_Len) -> freqs: (Batch, Seq_Len, Dim/2)
			current_freqs = self.freqs_cis[pos_ids]

		# Reshape freqs for broadcasting: (Batch, Seq_Len, Dim/2) -> (Batch, 1, Seq_Len, Dim/2)
		current_freqs = current_freqs.unsqueeze(1)

		# Apply rotation
		x_rotated = x_complex * current_freqs

		# Convert back to real
		x_out = torch.view_as_real(x_rotated).flatten(start_dim=-2)

		return x_out.type_as(x)

class FlashMultiHeadAttention(nn.Module):
	def __init__(self, d_model, hidden_dim, maxlen, num_heads, dropout_rate):
		super(FlashMultiHeadAttention, self).__init__()

		self.num_heads = num_heads
		self.head_dim = hidden_dim // num_heads
		self.dropout_rate = dropout_rate

		assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
		assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

		self.q_linear = nn.Linear(d_model, hidden_dim)
		self.k_linear = nn.Linear(d_model, hidden_dim)
		self.v_linear = nn.Linear(d_model, hidden_dim)
		self.out_linear = nn.Linear(hidden_dim, d_model)

		self.rope = RoPE(self.head_dim, maxlen)

	def forward(self, x: torch.Tensor, attn_mask = None, pos_ids = None, kv_cache: KVCache = None, valid_lengths: List[int] = None):
		"""
		x: (Batch, Seq_Len, D_Model)
		kv_cache: KVCache object for this specific layer
		"""
		batch_size, seq_len, _ = x.size()

		# 1. Project Q, K, V
		Q = self.q_linear(x)
		K = self.k_linear(x)
		V = self.v_linear(x)

		# Reshape to (Batch, Seq_Len, Heads, Dim) -> Transpose to (Batch, Heads, Seq_Len, Dim)
		Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
		V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

		# 2. RoPE & KV Cache
		if kv_cache is not None:
			assert pos_ids is not None, "pos_ids must be provided when using kv_cache"

			# Apply RoPE to new queries and keys using the offset
			Q = self.rope(Q, pos_ids)
			K = self.rope(K, pos_ids)

			# Update Cache & Fetch Full History
			kv_cache.update(K, V, pos_ids, valid_lengths)
			K, V = kv_cache.fetch()

			# Note: attn_mask is pre-computed in Transformer block to save time
			is_causal = False
		else:
			# Standard Training / No-Cache Mode
			# Apply RoPE starting from 0
			Q = self.rope(Q)
			K = self.rope(K)

			# In standard training, we use is_causal=True.
			is_causal = True
			attn_mask = None

		# 3. Attention Dispatch
		if hasattr(F, 'scaled_dot_product_attention'):
			attn_output = F.scaled_dot_product_attention(
				Q, K, V,
				dropout_p=self.dropout_rate if self.training else 0.0, 
				attn_mask=attn_mask,
				is_causal=is_causal
			)
		else:
			if is_causal:
				attn_mask: torch.Tensor = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
				attn_mask = attn_mask.tril().unsqueeze(0)

			scale = (self.head_dim) ** -0.5
			scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * scale

			scores.masked_fill_(attn_mask.logical_not(), -torch.inf)

			attn_weights = F.softmax(scores, dim=-1)
			attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
			attn_output = torch.matmul(attn_weights, V)

		# 4. Output Projection
		attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
		output = self.out_linear(attn_output)

		return output

class PointWiseFeedForward(nn.Module):
	def __init__(self, d_model, dropout_rate):
		super(PointWiseFeedForward, self).__init__()

		# SwiGLU
		hidden_dim = d_model * 8 // 3
		hidden_dim = (hidden_dim + 31) // 32 * 32
		self.up_linear = nn.Linear(d_model, hidden_dim)
		self.down_linear = nn.Linear(hidden_dim, d_model)
		self.gate = nn.Linear(d_model, hidden_dim)
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, inputs):
		gate = F.silu(self.gate(inputs))
		filtered = gate * self.up_linear(inputs)
		outputs = self.down_linear(self.dropout(filtered))
		return outputs

class Transformer(nn.Module):
	def __init__(self, d_model: int, max_seq_len: int, num_blocks = 4, num_heads = 1, norm_first = True, dropout_rate = 0.0):
		super(Transformer, self).__init__()

		self.norm_first = norm_first
		self.num_blocks = num_blocks
		self.d_model = d_model
		self.hidden_dim = 2 * d_model
		self.max_seq_len = max_seq_len
		self.num_heads = num_heads

		self.attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
		self.attn_layers = nn.ModuleList([
			FlashMultiHeadAttention(d_model, self.hidden_dim, max_seq_len, num_heads, dropout_rate)
			for _ in range(num_blocks)
		])
		self.fwd_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
		self.fwd_layers = nn.ModuleList([PointWiseFeedForward(d_model, dropout_rate) for _ in range(num_blocks)])

	@property
	def cache_shape(self):
		return torch.Size((self.num_heads, self.max_seq_len, self.hidden_dim // self.num_heads))

	def create_kv_cache(self, buffer: TensorBuffer):
		"""
		Creates a list of rows (one per block) for a single sequence.
		"""
		return [KVCacheRow(buffer) for _ in range(self.num_blocks)]

	def forward(self, seqs: torch.Tensor, kv_caches: List[KVCache] = None, valid_lengths: List[int] = None):
		"""
		seqs: (Batch, Seq_Len, D_Model)
		kv_caches: List of KVCache objects (one per layer)
		valid_lengths: List of actual new token counts per sequence
		"""
		attn_mask = None
		pos_ids = None

		# Pre-computation for Inference Mode
		if kv_caches is not None:
			_, seq_len, _ = seqs.shape
			device = seqs.device

			# 1. Get Starting Offsets (Shape: Batch)
			# We use layer 0's cache to determine offsets (all layers are synced)
			offsets = kv_caches[0].get_offsets()

			# 2. Calculate Pos IDs (Offsets + Current Sequence indices)
			# Shape: (Batch, Seq_Len)
			# pos_ids[b, i] = offset[b] + i
			pos_ids = offsets.unsqueeze(1) + torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0)

			# 3. Calculate EXACT Max Position for Masking
			# The buffer will be updated up to: offset + valid_len
			# Note: We use valid_lengths (actual tokens), NOT seq_len (padded tokens)
			max_pos = max(
				cache.curr_pos + length
				for cache, length in zip(kv_caches[0].caches, valid_lengths)
			)

			# 4. Construct Mask
			# Rows: Global Position (Batch, Seq_Len, 1)
			rows = pos_ids.unsqueeze(-1)
			# Cols: Buffer Index (1, 1, Max_Pos)
			cols = torch.arange(max_pos, device=device).view(1, 1, -1)

			# Mask: True if Col <= Row
			attn_mask = cols <= rows
			attn_mask = attn_mask.unsqueeze(1) # (Batch, 1, Seq_Len, Max_Pos)

		# --------------------------------------------------------------
		# Transformer Loop
		# --------------------------------------------------------------
		for i in range(self.num_blocks):
			# Select specific cache for this layer if available
			layer_cache = kv_caches[i] if kv_caches is not None else None

			if self.norm_first:
				x = self.attn_norms[i](seqs)
				# Pass pre-computed mask/pos_ids
				mha_outputs = self.attn_layers[i](x, attn_mask, pos_ids, layer_cache, valid_lengths)
				seqs = seqs + mha_outputs
				seqs = seqs + self.fwd_layers[i](self.fwd_norms[i](seqs))
			else:
				mha_outputs = self.attn_layers[i](seqs, attn_mask, pos_ids, layer_cache, valid_lengths)
				seqs = self.attn_norms[i](seqs + mha_outputs)
				seqs = self.fwd_norms[i](seqs + self.fwd_layers[i](seqs))

		return seqs

class Model(nn.Module):
	def __init__(self, n_toks: int, n_players: int, n_actions: int, d_model: int, max_seq_len: int, **conf):
		super(Model, self).__init__()

		self.n_players = n_players
		self.n_actions = n_actions
		self.max_seq_len = max_seq_len
		self.tok_emb = nn.Embedding(n_toks, d_model)
		self.player_emb = nn.Embedding(n_players + 1, d_model, padding_idx=0)
		self.policy_fn = nn.Linear(d_model, n_actions)
		self.value_fn = nn.Linear(d_model, 1)
		self.transformer = Transformer(d_model, max_seq_len, **conf)

		self.apply(init_weight)
	
	def forward(self, toks: torch.Tensor, id_toks: torch.Tensor, kv_caches: List[KVCache] = None, valid_lengths: List[int] = None):
		seqs = self.tok_emb(toks) + self.player_emb(id_toks)
		seqs = self.transformer(seqs, kv_caches, valid_lengths)
		logits = self.policy_fn(seqs)
		values = self.value_fn(seqs).squeeze(-1)
		return logits, values
	
	@torch.inference_mode()
	def get_action_and_value(
		self,
		toks: torch.Tensor,
		id_toks: torch.Tensor,
		action_mask: torch.Tensor,
		kv_caches: List[KVCache] = None,
		valid_lengths: List[int] = None,
		deterministic = False
	) -> Dict[str, torch.Tensor]:
		"""
		toks: (Batch, Seq_Len) - new tokens appended at last
		id_toks: (Batch, Seq_Len) - player id of each token
		action_mask: (Batch, N_Actions) - valid actions
		"""
		logits, values = self(toks, id_toks, kv_caches, valid_lengths)
		# Get the logits and values of the last valid position
		# logits: (Batch, Seq_Len, N_Actions) -> (Batch, N_Action)
		# values: (Batch, Seq_Len) -> (Batch,)
		logits = torch.stack([logits[i, length - 1] for i, length in enumerate(valid_lengths)])
		logits.masked_fill_(action_mask.logical_not(), -torch.inf)
		logits -= logits.logsumexp(dim=-1, keepdim=True)
		values = torch.stack([values[i, length - 1] for i, length in enumerate(valid_lengths)])
		if deterministic:
			actions = logits.argmax(dim=-1, keepdim=True)
		else:
			probs = torch.exp(logits)
			actions = torch.multinomial(probs, num_samples=1, replacement=True)
		log_probs = logits.gather(-1, actions).squeeze(dim=-1)
		actions = actions.squeeze(dim=-1)
		return {
			'actions': actions,
			'logits': logits,
			'log_probs': log_probs,
			'values': values,
		}
