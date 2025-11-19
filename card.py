from typing import Tuple

Card = Tuple[int, int, int]

ID_TO_SUIT = ['s', 'h', 'c', 'd']
SUIT_TO_ID = {'': None, 'n': -1, 's': 0, 'h': 1, 'c': 2, 'd': 3}
NUMBER_TO_ID = {k: v for v, k in enumerate([
	'2','3','4','5','6','7','8','9','10','J','Q','K','A'
])}
