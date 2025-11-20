from agent import Agent
from policy import random_policy
from game import Tractor

class Env:
	def __init__(self):
		self.players = [Agent() for _ in range(4)]
		self.game = Tractor()
		self.req = self.game.reset()

	def step(self):
		stage = self.req['stage']
		player_id = self.req['playerpos']
		player = self.players[player_id]
		player.observe(self.req)

		if stage == 'deal':
			toks, mask = player.obs()
			tok = random_policy(mask)
			ids =  player.tok_to_ids(tok)

		elif stage == 'cover':
			ids = []
			for _ in range(8):
				toks, mask = player.obs()
				tok = random_policy(mask)
				ids += player.tok_to_ids(tok)

			for id in ids:
				player.hand.remove(id)

		elif stage == 'play':
			ids = []
			while True:
				toks, mask = player.obs()
				tok = random_policy(mask)
				new_ids = player.tok_to_ids(tok)
				if new_ids is None:
					break
				else:
					ids += new_ids

		else:
			raise NotImplementedError(f"{stage}: unknown stage")

		response = {
			'player': player_id,
			'action': ids,
		}
		req, reward, done = self.game.step(response)
		self.req = req
		if done:
			print(reward)
		return done

if __name__ == '__main__':
	while True:
		env = Env()
		while True:
			done = env.step()
			if done:
				break
