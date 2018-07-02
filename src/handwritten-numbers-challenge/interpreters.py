import numpy as np

# interpreter functions of output layer

class ArgMax:

    def val(self, x):
        return np.argmax(x)

    def inv(self, num_out, z):
        ''' Compute an inverse of z. '''
        ret = np.zeros((num_out,1))
        ret[z] = 1.0
        return ret
