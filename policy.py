import numpy as np

def random_policy(mask: np.ndarray):
	print(f'{mask.sum()=}')
	idxs = np.flatnonzero(mask)
	return np.random.choice(idxs)
