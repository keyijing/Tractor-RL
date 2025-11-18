from state import Stage
from agent import Agent
from policy import random_policy
from judger import main
import json

class Env:
	def __init__(self):
		self.players = [Agent() for _ in range(4)]
		self.logs = {'log': []}

	def step(self):
		output = main(self.logs)
		if output['command'] != 'request':
			print(output)
			return True

		if 'initdata' in output:
			initdata = output.pop('initdata')
			self.logs['initdata'] = initdata
			print(json.dumps({'initdata': initdata}))
		self.logs['log'].append({'output': output})

		content = output['content']
		print(json.dumps(content))

		player_id, request = next(iter(content.items()))
		player = self.players[int(player_id)]
		stage = request['stage']
		player.observe(request)
		print(player.state.hand)

		if stage == 'deal':
			stage, mask = player.obs()
			assert stage == Stage.DEAL

			tok = random_policy(mask)
			ids =  player.tok_to_ids(tok)

		elif stage == 'cover':
			ids = []
			for _ in range(8):
				stage, mask = player.obs()
				assert stage == Stage.COVER

				tok = random_policy(mask)
				ids += player.tok_to_ids(tok)

			print(ids)
			for id in ids:
				player.state.hand.remove(id)

		elif stage == 'play':
			ids = []
			while True:
				stage, mask = player.obs()
				assert stage == Stage.PLAY

				tok = random_policy(mask)
				new_ids = player.tok_to_ids(tok)
				if new_ids is None:
					break
				else:
					ids += new_ids
					print(new_ids)

		else:
			raise NotImplementedError(f"{stage}: unknown stage")

		response = {
			player_id: {'response': ids}
		}
		print(json.dumps(response))
		self.logs['log'].append(response)
		return False

if __name__ == '__main__':
	env = Env()
	while True:
		done = env.step()
		if done:
			break
