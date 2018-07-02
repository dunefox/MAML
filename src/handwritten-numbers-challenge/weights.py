import numpy as np

# weights

class NoWeight:

    def val(self, W):
        return 0

    def diff(self, W):
        return 0

class L1:

    def val(self, W):
        return np.abs(W).sum() / W.size

    def diff(self, W):
        return np.sign(W) / W.size

class L2:

    def val(self, W):
        return 0.5 * np.square(W).sum() / W.size

    def diff(self, W):
        return W / W.size

