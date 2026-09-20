import numpy as np

class MarkovChain(object):
    def __init__(self, transmat, x0=None):
        self.transmat = transmat
        self.n_states = transmat.shape[0]
        self.x0 = np.random.randint(self.n_states) if x0 is None else x0
        self.transmat_rowsum = np.cumsum(transmat, axis=1)

    def draw(self, xprev=None):
        xprev = self.x0 if xprev is None else xprev
        return np.sum(np.random.uniform() > self.transmat_rowsum[xprev, :])

    def draw_path(self, xprev=None, n_obs=1):
        xprev = self.x0 if xprev is None else xprev
        x_path = [xprev,]
        for ii in range(n_obs):
            x_path.append(self.draw(x_path[-1]))
        return np.array(x_path)
