from card import SUIT_TO_ID, NUMBER_TO_ID, id_to_card, card_to_tok
from tractor import DealState, CoverState, PlayState
from enum import Enum
from itertools import count
import numpy as np

class Stage(Enum):
	DEAL = 0
	COVER = 1
	PLAY = 2

CARD_TOK  = 0    #   0 ~ 53  (54)
PLAY_OUT  = 54   #  54 ~ 163 (108 + 2)
SCORE_TOK = 164  # 164 ~ 170 (0 ~ 2, 3, 4 ~ 7, 8, 9 ~ 10, 11, 12)
MAJOR_TOK = 171  # 171 ~ 175 (4 + 1)
COVER_TOK = 176
TRICK_TOK = 177
N_TOKENS  = 178

DEAL_MASK  = 0   #  0 ~ 10  (10 + 1)
COVER_MASK = 5   # 11 ~ 64  (54)
PLAY_MASK  = 59  # 65 ~ 173 (108 + 1)
N_ACTIONS  = 174

class Agent:
	def __init__(self):
		self.player_id = None

		# game state
		self.level = None
		self.major = None
		self.caller = -1
		self.snatcher = -1
		self.major = None
		self.banker = -1

		# cards currently held
		self.hand: list[int] = []

		# current stage
		self.deal = None
		self.cover = None
		self.play = None

		# seq toks
		self.toks = []

	def _update_banking(self, banking):
		caller = self.get_id(banking['called'])
		snatcher = self.get_id(banking['snatched'])
		major = SUIT_TO_ID[banking['major']]
		banker = self.get_id(banking['banker'])

		if major != self.major:
			self.major = major

			if banker != self.banker:
				self.toks.append((MAJOR_TOK + major + 1, banker + 1))
			elif snatcher != self.snatcher:
				self.toks.append((MAJOR_TOK + major + 1, snatcher + 1))
			elif caller != self.caller:
				self.toks.append((MAJOR_TOK + major + 1, caller + 1))
			else:
				self.toks.append((MAJOR_TOK + major + 1, 0))

		self.caller = caller
		self.snatcher = snatcher
		self.banker = banker

	def _get_tok(self, id: int, player: int):
		return CARD_TOK + card_to_tok(id_to_card(id, self.level)), player + 1

	def observe(self, req):
		stage = req['stage']

		if self.player_id is None:
			self.player_id = req['playerpos']
			self.get_id = lambda id: -1 if id == -1 else (id - self.player_id) % 4

		glob = req['global']
		if self.level is None:
			level = glob['level']
			self.level = NUMBER_TO_ID[level]

			self.toks.append((SCORE_TOK + (
				0 if self.level <= 2 else
				1 if self.level == 3 else
				2 if self.level <= 7 else
				3 if self.level == 8 else
				4 if self.level <= 10 else
				5 if self.level == 11 else
				6
			), 0))

		banking = glob['banking']
		self._update_banking(banking)

		if stage == 'deal':
			self.stage = Stage.DEAL
			id = req['deliver'][0]
			self.hand.append(id)

			self.toks.append(self._get_tok(id, -1))

		elif stage == 'cover':
			self.stage = Stage.COVER
			cards = req['deliver']
			self.hand.extend(cards)

			self.toks.extend([self._get_tok(id, -1) for id in cards])
			self.toks.append((COVER_TOK, 0))

		elif stage == 'play':
			self.stage = Stage.PLAY

			flag = False
			for cards, player in zip(req['history'][0], count(req['history'][2])):
				player_id = self.get_id(player)
				if player_id == 0:
					flag = True
					for card in cards:
						self.hand.remove(card)
				if flag:
					self.toks.extend([self._get_tok(id, player_id) for id in cards])

			self.toks.append((TRICK_TOK, 0))

			current = req['history'][1]
			self.leading = current[0] if current else []
			for cards, player in zip(current, count(req['history'][3])):
				player_id = self.get_id(player)
				self.toks.extend([self._get_tok(id, player_id) for id in cards])

		else:
			raise NotImplementedError(f"{stage}: unknown stage")

	def obs(self):
		mask = np.zeros(N_ACTIONS, dtype=bool)

		if self.stage == Stage.DEAL:
			if self.deal is None:
				self.deal = DealState(self.level)

			self.deal.action_mask(self.hand[-1], self.caller, self.snatcher, mask[DEAL_MASK : DEAL_MASK+11])

		elif self.stage == Stage.COVER:
			if self.cover is None:
				self.cover = CoverState(self.level, self.hand)

			self.cover.action_mask(mask[COVER_MASK : COVER_MASK+54])

		elif self.stage == Stage.PLAY:
			if self.play is None:
				self.play = PlayState(self.level, self.major, self.hand, self.leading)

			self.play.action_mask(mask[PLAY_MASK : PLAY_MASK+110])

		else:
			raise NotImplementedError()

		toks = self.toks
		self.toks = []
		return toks, mask

	def tok_to_ids(self, tok: int):
		if self.stage == Stage.DEAL:
			tok -= DEAL_MASK
			return self.deal.tok_to_ids(tok)

		elif self.stage == Stage.COVER:
			tok -= COVER_MASK
			return self.cover.tok_to_ids(tok)

		elif self.stage == Stage.PLAY:
			tok -= PLAY_MASK
			self.toks.append((PLAY_OUT + tok, 0))

			if tok == self.play.eos_tok:
				self.play = None
				return None
			else:
				return self.play.tok_to_ids(tok)

		else:
			raise NotImplementedError()
