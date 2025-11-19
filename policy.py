from state import Stage
import numpy as np

def random_policy(mask: np.ndarray):

	if mask[-1] and np.random.random() < 0.5:
		# kill
		return len(mask) - 1

	idxs = np.flatnonzero(mask)
	return np.random.choice(idxs)
