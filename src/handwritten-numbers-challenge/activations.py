import numpy as np

# activation functions and their derivatives 

class Sigmoid:

    def val(self, y):
        return 1.0 / ( 1.0 + np.exp(-y) )

    def diff(self, y):
        a = self.val(y)
        return a * ( 1 - a ) * np.identity(a.size)

class Tanh:

    def val(self, y):
        return np.tanh(y) 

    def diff(self, y):
        return 1 / np.square( np.cosh(y) ) 

class Relu:

    def val(self, y):
        return np.maximum(np.zeros(y.shape), y)

    def diff(self, y):
        return ( 1.0 + np.sign(y) ) / 2.0 * np.identity(y.size) 

class Linear:

    def val(self, y):
        return y

    def diff(self, y):
        return np.identity(y.size)

class Softmax:

    def val(self, y):
        dy = y - y.max()
        e = np.exp(dy)
        return e / e.sum()

    def diff(self, y):
        a = self.val(y)
        return np.diagflat(a) - np.dot(a, a.T)
