from typing import Tuple

Card = Tuple[int, int, int]

SUIT_TO_ID = {'': None, 'n': -1, 'h': 0, 'd': 1, 's': 2, 'c': 3}
NUMBER_TO_ID = {k: v for v, k in enumerate([
	'2','3','4','5','6','7','8','9','10','J','Q','K','A'
])}
