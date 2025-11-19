from state import AgentState, Stage
from tractor import DealState, CoverState, PlayState
import numpy as np

class Agent:
	def __init__(self):
		self.state = AgentState()
		self.deal = None
		self.cover = None
		self.play = None

	def observe(self, req):
		self.state.observe(req)

	def obs(self):
		state = self.state

		if state.stage == Stage.DEAL:
			if self.deal is None:
				self.deal = DealState(state.level)

			mask = np.zeros(12, dtype=bool)
			self.deal.action_mask(state.hand[-1], state.caller, state.snatcher, mask)

			return Stage.DEAL, mask

		elif state.stage == Stage.COVER:
			if self.cover is None:
				self.cover = CoverState(state.level, state.hand)

			mask = np.zeros(54, dtype=bool)
			self.cover.action_mask(mask)

			return Stage.COVER, mask

		elif state.stage == Stage.PLAY:
			if self.play is None:
				leading = state.current[0][1] if state.current else []
				self.play = PlayState(state.level, state.major, state.hand, leading)

			mask = np.zeros(110, dtype=bool)
			self.play.action_mask(mask)

			return Stage.PLAY, mask

		else:
			raise NotImplementedError()

	def tok_to_ids(self, tok: int):
		stage = self.state.stage

		if stage == Stage.DEAL:
			return self.deal.tok_to_ids(tok)

		elif stage == Stage.COVER:
			return self.cover.tok_to_ids(tok)

		elif stage == Stage.PLAY:
			if tok == self.play.eos_tok:
				self.play = None
				return None
			else:
				return self.play.tok_to_ids(tok)

		else:
			raise NotImplementedError()
