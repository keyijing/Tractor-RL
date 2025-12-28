import sys, os
sys.path.append('/data/Tractor')

from agent import N_TOKENS, N_ACTIONS, Stage, Agent
from model import Model
import numpy as np
import torch
import json

_online = os.environ.get("USER", "") == "root"
if _online:
	full_input: dict = json.loads(input())
else:
	with open("log_forAI.json") as f:
		full_input: dict = json.load(f)

# loading model
device = 'cpu'
model = Model(
	n_toks=N_TOKENS,
	n_players=4,
	n_actions=N_ACTIONS,
	d_model=256,
	max_seq_len=384,
	num_blocks=16,
	num_heads=8,
)
data_dir = '/data/Tractor/model.pt'
model.load_state_dict(torch.load(data_dir, map_location=device))
model.to(device)
model.eval()

player = Agent()
his_toks = []
res_toks: list = full_input.get('data', [])
cur_toks = res_toks

def policy(toks: list, action_mask: np.ndarray):
	global res_toks
	global cur_toks

	if cur_toks:
		tok = cur_toks[0]
		cur_toks = cur_toks[1:]
		return tok

	toks_tensor = torch.tensor(toks, dtype=torch.int64, device=device).unsqueeze(dim=0)
	action_mask_tensor = torch.from_numpy(action_mask).to(device).unsqueeze(dim=0)
	output = model.get_action_and_value(
		toks=toks_tensor[..., 0],
		id_toks=toks_tensor[..., 1],
		action_mask=action_mask_tensor,
		valid_lengths=[len(toks)],
		deterministic=True,
	)
	tok = output['actions'][0].item()
	res_toks.append(tok)
	return tok

for req in full_input['requests']:
	player.observe(req)

	if player.stage == Stage.DEAL:
		toks, mask = player.obs()
		his_toks += toks
		tok = policy(his_toks, mask)
		ids = player.tok_to_ids(tok)

	elif player.stage == Stage.COVER:
		ids = []
		for _ in range(8):
			toks, mask = player.obs()
			his_toks += toks
			tok = policy(his_toks, mask)
			ids += player.tok_to_ids(tok)
		for id in ids:
			player.hand.remove(id)

	elif player.stage == Stage.PLAY:
		ids = []
		while True:
			toks, mask = player.obs()
			his_toks += toks
			tok = policy(his_toks, mask)
			new_ids = player.tok_to_ids(tok)
			if new_ids is None:
				break
			else:
				ids += new_ids

print(json.dumps({
	'response': ids,
	'data': res_toks,
}))
