from typing import Tuple

Card = Tuple[int, int, int]

ID_TO_SUIT = ['s', 'h', 'c', 'd']
SUIT_TO_ID = {'': None, 'n': -1, 's': 0, 'h': 1, 'c': 2, 'd': 3}
NUMBER_TO_ID = {k: v for v, k in enumerate([
	'2','3','4','5','6','7','8','9','10','J','Q','K','A'
])}

def tok_to_card(tok: int) -> Card:
	if tok < 52:
		number, suit = divmod(tok, 4)
		return 0 if tok < 48 else 1, suit, number
	else:
		return 2, 0, tok - 52

def card_to_tok(card: Card):
	type, suit, number = card
	return 4 * number + suit if type <= 1 else 52 + number

def id_to_card(id: int, level: int) -> Card:
	"""
	Output: (type, suit, number)
	  type - 2=joker, 1=level card, 0=normal
	"""
	id %= 54
	if id >= 52:
		return 2, 0, id - 52
	number, suit = divmod(id, 4)
	number = (number - 1) % 13
	if number == level:
		number = 12
	elif number > level:
		number -= 1
	return 1 if number == 12 else 0, suit, number

def card_to_id(card: Card, level: int):
	type, suit, number = card
	if type == 2:
		return 52 + number
	if number == 12:
		number = level
	elif number >= level:
		number += 1
	number = (number + 1) % 13
	return 4 * number + suit
