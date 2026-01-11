from agent import Agent
from game import Tractor

class Env:
	def __init__(self, seed = None):
		self.game = Tractor(seed)

	def reset(self, level=None):
		self.players = [Agent() for _ in range(4)]
		self.gains = [[0, 0] for _ in range(4)] # (reward, punish)
		self.req = self.game.reset(level)
		self.done = False

	def obs(self):
		if self.req:
			self.stage = self.req['stage']
			self.player_id = int(self.req['playerpos'])
			self.player = self.players[self.player_id]
			self.ids = []

			self.player.observe(self.req)
			self.req = None

		toks, mask = self.player.obs()
		gain = self.gains[self.player_id]
		self.gains[self.player_id] = [0, 0]
		return {
			'player': self.player_id,
			'reward': gain[0],
			'punish': gain[1],
			'toks': toks,
			'action_mask': mask,
		}

	def step(self, tok: int):
		if self.stage == 'deal':
			self.ids = self.player.tok_to_ids(tok)
			next_step = True

		elif self.stage == 'cover':
			self.ids += self.player.tok_to_ids(tok)
			if len(self.ids) == 8:
				for id in self.ids:
					self.player.hand.remove(id)
				next_step = True
			else:
				next_step = False

		elif self.stage == 'play':
			new_ids = self.player.tok_to_ids(tok)
			if new_ids is None:
				next_step = True
			else:
				self.ids += new_ids
				next_step = False

		if next_step:
			response = {
				'player': self.player_id,
				'action': self.ids,
			}
			self.req, gains, self.done = self.game.step(response)
			for i, (reward, punish) in enumerate(gains):
				self.gains[i][0] += reward
				self.gains[i][1] += punish
			return self.done
		else:
			return False
