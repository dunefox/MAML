import numpy as np

# objective functions and their derivatives

class L2:

    def val(self, xN, xBar):
        return .5 * np.square(xN - xBar).sum()
    
    def diff(self, xN, xBar):
        return (xN - xBar).T

class CrossEntropy:

    def val(self, xN, xBar):
        return - np.nan_to_num( np.dot(xBar.T, np.log(xN)) + np.dot((1-xBar).T, np.log(1-xN)) )
    
    def diff(self, xN, xBar):
        return np.nan_to_num( ( (xN - xBar) / (xN * (1-xN) ) ).T )
