from state import Stage
import numpy as np

def random_policy(stage: Stage, mask: np.ndarray):
	# print(f'{mask.sum()=}')

	if stage == Stage.PLAY and mask[108] and np.random.random() < 0.5:
		# print("==== kill ====")
		return 108

	idxs = np.flatnonzero(mask)
	return np.random.choice(idxs)
