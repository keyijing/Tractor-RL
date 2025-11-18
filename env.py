from state import Stage
from agent import Agent
from policy import random_policy
import judger
import json
from copy import deepcopy
from importlib import reload

class Env:
	def __init__(self):
		self.players = [Agent() for _ in range(4)]
		self.log = []
		self.initdata = None

	def step(self):
		logs = {'log': deepcopy(self.log)}
		if self.initdata is not None:
			logs['initdata'] = deepcopy(self.initdata)
		reload(judger)
		output = judger.main(logs)
		if output['command'] != 'request':
			print(output)
			return True

		if 'initdata' in output:
			self.initdata = output.pop('initdata')
		self.log.append({'output': output})

		content = output['content']
		player_id, request = next(iter(content.items()))
		player = self.players[int(player_id)]
		stage = request['stage']
		player.observe(request)

		if stage == 'deal':
			stage, mask = player.obs()
			assert stage == Stage.DEAL

			tok = random_policy(Stage.DEAL, mask)
			ids =  player.tok_to_ids(tok)

		elif stage == 'cover':
			ids = []
			for _ in range(8):
				stage, mask = player.obs()
				assert stage == Stage.COVER

				tok = random_policy(Stage.COVER, mask)
				ids += player.tok_to_ids(tok)

			for id in ids:
				player.state.hand.remove(id)

		elif stage == 'play':
			ids = []
			while True:
				stage, mask = player.obs()
				assert stage == Stage.PLAY

				tok = random_policy(Stage.PLAY, mask)
				new_ids = player.tok_to_ids(tok)
				if new_ids is None:
					break
				else:
					ids += new_ids

		else:
			raise NotImplementedError(f"{stage}: unknown stage")

		response = {
			player_id: {'response': ids}
		}
		self.log.append(response)
		return False

if __name__ == '__main__':
	while True:
		env = Env()
		while True:
			done = env.step()
			if done:
				break
