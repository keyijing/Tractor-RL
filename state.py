from card import Card, SUIT_TO_ID, NUMBER_TO_ID
from tractor import id_to_card
from enum import Enum
from itertools import count

class Stage(Enum):
	DEAL = 0
	COVER = 1
	PLAY = 2

class AgentState:
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

		# list of finished tricks
		self.history: list[list[list[Card]]] = []

		# list of (player, card) for current trick
		self.current: list[list[Card]] = []

	def observe(self, req):
		stage = req['stage']

		if self.player_id is None:
			self.player_id = req['playerpos']
			self.get_id = lambda id: -1 if id == -1 else (id - self.player_id) % 4

		glob = req['global']
		if self.level is None:
			level = glob['level']
			self.level = NUMBER_TO_ID[level]

		banking = glob['banking']
		caller = self.get_id(banking['called'])
		snatcher = self.get_id(banking['snatched'])
		major = SUIT_TO_ID[banking['major']]
		banker = self.get_id(banking['banker'])

		if caller != self.caller:
			self.caller = caller
		if snatcher != self.snatcher:
			self.snatcher = snatcher
		if major != self.major:
			self.major = major
		if banker != self.banker:
			self.banker = banker

		if stage == 'deal':
			self.stage = Stage.DEAL
			self.hand.extend(req['deliver'])

		elif stage == 'cover':
			self.stage = Stage.COVER
			self.hand.extend(req['deliver'])

		elif stage == 'play':
			self.stage = Stage.PLAY
			last_trick = []
			for player, cards in zip(count(req['history'][2]), req['history'][0]):
				if self.get_id(player) == 0:
					for card in cards:
						self.hand.remove(card)

				cards = [id_to_card(id, self.level) for id in cards]
				last_trick.append((self.get_id(player), cards))

			if last_trick:
				# process last_trick

				self.history.append(last_trick)

			self.current = []
			for player, cards in zip(count(req['history'][3]), req['history'][1]):
				cards = [id_to_card(id, self.level) for id in cards]
				self.current.append((self.get_id(player), cards))

			# process current_trick

		else:
			raise NotImplementedError(f"{stage}: unknown stage")
