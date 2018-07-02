import time
import math
import random
import numpy as np

import interpreters as interpreters
import objectives as objectives

class NN:
    ''' 
    Multi-layer perceptron class.

    :param num_in: Number of input neurons.
    :param interpret: `nninterprets` class which handles the interpretation of the
        activation in the output layer.
    :param objective: nconsts class which defines the objective function that should be used.

    .. code-block: python
        network = NN( 30, ArgMax, L2 )

    '''

    def __init__(self, num_in, 
                 interpret=interpreters.ArgMax, 
                 objective=objectives.L2):

        self.n = [ num_in ] # list of total number of neurons in each layer
        self.layers = [] # list of neuron layers
        self.interpret = interpret() # interpreter function for the output neurons
        self.objective = objective() # objective function of network

        return

    def add_layer(self, 
                  num, 
                  layer, 
                  **args):
        ''' 
        Add another layer of `num` neurons of type `layer` and activation
        function `alpha`. 

        :param num: Number of neurons in the added layer.
        :param layer: `nnlayers` class which provides the logic of the added layer.
        :param alpha: `nnactivations` class which provides the activation function of the added layer.
        '''

        self.layers.append(layer(self.n[-1], num, args))
        self.n.append(num)

        return 

    def feed_forward(self, x):
        ''' 
        Compute output neurons for input `x`.  Output is interpreted by given
        function `iterpret` which is the argmax by default. 
        '''

        xout = x
        for l in self.layers:
            xout = l.val(xout)

        return xout, self.interpret.val(xout)

    def train_batch(self, data):

        # reset layer parameters to be able to add the computed deltas
        # in `add_dxk_over_dpk` and average them out in `update_params`
        for l in self.layers:
            l.reset_delta_params()

        for x0, xBar_label in data:

            # training label
            xBar = self.interpret.inv(self.n[-1], xBar_label)

            # feed forward
            xN, _ = self.feed_forward(x0)

            # feed backwards
            product = self.objective.diff(self.layers[-1].x, xBar)
            for i in reversed(range(1, len(self.layers))):
                self.layers[i].add_dxk_over_dpk(product, self.layers[i-1].x)
                product = np.dot( product, self.layers[i].dxk_over_dxkm1() )
            self.layers[0].add_dxk_over_dpk(product, x0)

        for l in self.layers:
            l.update_params(len(data))

        return 

    def test_network(self, data):
        ''' Compute objective and list of errors. '''
            
        objective = 0.0
        error_list = []
        for i in range(len(data)):

            img = data[i][0]
            label = data[i][1]
            
            x0 = img
            xN, predict = self.feed_forward(x0)
            xBar = self.interpret.inv(self.n[-1], label)
            
            objective += self.objective.val(xN, xBar)
            if predict != label:
                error_list.append((i, label, predict))
        
        return objective / len(data),  error_list

    def epoch_monitor(self, **kwargs):
        ''' 
        Monitor progress in each epoch. To be called from batch_training(...).
        '''
        
        epoch_dts = kwargs['epoch_dts']
        batch_dts = kwargs['batch_dts'] 
        epochs = kwargs['epochs']
        batches = kwargs['batches']
        batch_size = kwargs['batch_size']
        train_data = kwargs['train_data']
        test_data = kwargs['test_data']
        
        epoch_num = len(epoch_dts)
        epoch_progress = float(epoch_num) / epochs
        
        print('> Epoch=%d/%d (%.2f%%), Dt=%fs, Avg=%fs'
              % (epoch_num, epochs, epoch_progress * 100, epoch_dts[-1], np.mean(epoch_dts)))

        return

    def batch_monitor(self, **kwargs):
        ''' 
        Monitor progress in each batch. To be called from batch_training(...).
        '''
        
        epoch_dts = kwargs['epoch_dts']
        batch_dts = kwargs['batch_dts'] 
        epochs = kwargs['epochs']
        batches = kwargs['batches']
        batch_size = kwargs['batch_size']
        train_data = kwargs['train_data']
        test_data = kwargs['test_data']
        
        epoch_num = len(epoch_dts)
        epoch_progress = float(epoch_num) / epochs
        batch_num = len(batch_dts)
        batch_progress = float(batch_num * batch_size) / len(train_data)

        if epoch_num == 0:
            eta = ( epochs * batches ) * np.mean(batch_dts) 
        else:
            eta = (epochs - epoch_num) * np.mean(epoch_dts) \
                    + (batches - batch_num) * np.mean(batch_dts) 

        print('| Epoch=%d, Batch=%d/%d (%.2f%%), Dt=%fs, Avg=%fs, ETA: %fs'
              % (epoch_num, batch_num, batches, batch_progress * 100, 
                 batch_dts[-1], np.mean(batch_dts), eta), end='\r')
        
        if batch_num == batches:
            print()

        return

    def batch_training(self, train_data, test_data, 
                       epochs, batch_size, 
                       **kwargs):
        ''' Prepare batches and train those over given number of epochs. '''

        batches = math.ceil(float(len(train_data)) / batch_size)
        # loop through epochs
        epoch_dts = []
        for e in range(epochs):
           
            epoch_t0 = time.time()

            # generate radom indices to shuffle data
            l = len(train_data)
            indices = random.sample(range(l),l)

            # loop through all batches
            batch_dts = []
            ptr_first = 0
            while ptr_first < len(indices):

                # prepare batch pointers
                ptr_last = min(len(train_data), ptr_first + batch_size)
                idxs = indices[ptr_first:ptr_last]

                batch_t0 = time.time()
                self.train_batch([train_data[i] for i in idxs])
                batch_t1 = time.time()
                batch_dts.append(batch_t1 - batch_t0)

                # point to next batch
                ptr_first += batch_size

                # call monitor function
                self.batch_monitor(epoch_dts=epoch_dts, batch_dts=batch_dts, 
                                   epochs=epochs, batches=batches, batch_size=batch_size, 
                                   train_data=train_data, test_data=test_data, 
                                   kwargs=kwargs)

            # status of current epoch
            epoch_t1 = time.time()
            epoch_dts.append(epoch_t1 - epoch_t0) 

            # call monitor function
            self.epoch_monitor(epoch_dts=epoch_dts, batch_dts=batch_dts, 
                               epochs=epochs, batches=batches, batch_size=batch_size, 
                               train_data=train_data, test_data=test_data, 
                               kwargs=kwargs)
     
        return
