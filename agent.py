from state import AgentState, Stage
from tractor import DealState, CoverState, PlayState
import numpy as np

class Agent:
	def __init__(self):
		self.state = AgentState()
		self.cover_count = 0
		self.play = None

	def observe(self, req):
		self.state.observe(req)

	def obs(self):
		state = self.state

		if state.stage == Stage.DEAL:
			self.deal = DealState(state.level, state.caller, state.snatcher, state.hand)

			mask = np.zeros(12, dtype=bool)
			self.deal.action_mask(mask)

			return Stage.DEAL, mask

		elif state.stage == Stage.COVER:
			if self.cover_count == 0:
				self.cover = CoverState(state.level, state.hand)

			mask = np.zeros(54, dtype=bool)
			self.cover.action_mask(mask)

			return Stage.COVER, mask

		elif state.stage == Stage.PLAY:
			if self.play is None:
				leading = state.current[0] if state.current else []
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
			self.cover_count = (self.cover_count + 1) % 8
			return self.cover.tok_to_ids(tok)

		elif stage == Stage.PLAY:
			if tok == self.play.eos_tok:
				return None
			else:
				return self.play.tok_to_ids(tok)

		else:
			raise NotImplementedError()
