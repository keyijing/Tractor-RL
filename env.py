from agent import Agent
from policy import random_policy
from game import Tractor

class Env:
	def __init__(self):
		self.game = Tractor()

	def reset(self):
		self.players = [Agent() for _ in range(4)]
		self.rewards = [0 for _ in range(4)]
		self.req = self.game.reset()

		self.sum = 0

	def obs(self):
		if self.req:
			self.stage = self.req['stage']
			self.player_id = self.req['playerpos']
			self.player = self.players[self.player_id]
			self.ids = []

			self.player.observe(self.req)
			self.req = None

		toks, mask = self.player.obs()
		reward = self.rewards[self.player_id]
		self.rewards[self.player_id] = 0
		return {
			'player': self.player_id,
			'reward': reward,
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
			self.req, rewards, done = self.game.step(response)
			for i, reward in enumerate(rewards):
				self.rewards[i] += reward
			if done:
				print(self.rewards)
			return done
		else:
			return False

if __name__ == '__main__':
	env = Env()
	while True:
		env.reset()
		while True:
			obs = env.obs()
			tok = random_policy(obs['action_mask'])
			done = env.step(tok)
			if done:
				break
