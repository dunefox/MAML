import numpy as np

import activations as activations
import weights as weights

# connection functions and their derivatives

class FullyConnected:

    def __init__(self, num_in, num_out, args):

        # number of in and out neurons
        self.num_in = num_in
        self.num_out = num_out

        # set activation function
        if 'activation' in args:
            self.activation = args['activation']()
        else:
            self.activation = activations.Sigmoid()

        # set learning rate
        if 'eta' in args:
            self.eta = args['eta']
        else:
            # some default value
            self.eta = 0.1

        # set parameter weight function
        if 'weight' in args:
            self.params_weight_funct = args['weight']()
        else:
            # use L2 weight as default
            self.params_weight_funct = weights.L2()

        # set matrix learning rate
        if 'weight' in args:
            self.params_omega = args['omega']
        else:
            # some default value
            self.params_omega = 0.1

        # initialize weights to zero
        self.W = np.zeros((num_out, num_in))
        self.b = np.zeros((num_out, 1))
        # or according to the choices
        if 'init' in args:
            if args['init'] == "norm":
                # initialize weights with normal random numbers
                self.W = np.random.randn(num_out, num_in)
                self.b = np.random.randn(num_out, 1)
            elif args['init'] == "norm_adapt":
                # initialize weights with normal random numbers
                # but reduce the variance in order not to saturate
                # the neurons
                self.W = np.random.randn(num_out, num_in) / np.sqrt(num_in)
                self.b = np.random.randn(num_out, 1)

        self.delta_W = np.zeros_like(self.W)
        self.delta_b = np.zeros_like(self.b)
        self.da = np.zeros( (self.b.size, self.b.size) )

        return

    def reset_delta_params(self):
        self.delta_W = np.zeros_like(self.W)
        self.delta_b = np.zeros_like(self.b)
        return

    def val(self, x):
        self.y = np.dot(self.W, x) + self.b
        self.x = self.activation.val(self.y)
        return self.x

    def dxk_over_dxkm1(self):
        self.da = self.activation.diff(self.y)
        return np.dot( self.da, self.W )

    def add_dxk_over_dpk(self, p, xkm1): 
        prod = np.dot(p, self.activation.diff(self.y))
        self.delta_W = self.delta_W + np.dot( prod.T, xkm1.T )
        self.delta_b = self.delta_b + prod.T
        return 

    def update_params(self, reps):
        avg_W = self.delta_W / reps
        avg_b = self.delta_b / reps
        self.W = self.W - (self.params_omega * self.params_weight_funct.diff(self.W) + self.eta * avg_W )
        self.b = self.b - self.eta * avg_b
        return
