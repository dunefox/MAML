import numpy as np

# for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# neural network classes
import multilayernet
import objectives
import weights
import interpreters
import activations
import layers
import mnist

# add some function sto the classes that help with the analysis

class FullyConnected_Diag(layers.FullyConnected):
    ''' Fully connected layer with diagnosis functions.  '''

    def params_weight(self):
        ''' Return parameter weight of matrix W. '''

        return self.params_weight_funct.val(self.W)

    def get_params(self):
        ''' Return matrix W. '''

        return self.W

class NN_Diag(multilayernet.NN):
    ''' Neural network with diagnosis functions. '''

    def sum_params_weight(self):
        ''' Sum parameter weights of all layers. '''

        c = 0
        for l in self.layers:
            c += l.params_weight()
        return c
    
    def epoch_monitor(self, **kwargs):

        epoch_dts = kwargs['epoch_dts']
        batch_dts = kwargs['batch_dts'] 
        epochs = kwargs['epochs']
        batches = kwargs['batches']
        batch_size = kwargs['batch_size']
        train_data = kwargs['train_data']
        test_data = kwargs['test_data']
         
        test_objective, test_errors = self.test_network(test_data)
        test_eff = 1 - len(test_errors) / len(test_data)
        
        train_objective, train_errors = self.test_network(train_data)
        train_eff = 1 - len(train_errors) / len(train_data)
        
        params_weight = self.sum_params_weight()

        epoch_num = len(epoch_dts)
        epoch_progress = float(epoch_num) / epochs

        print('> Epoch=%d/%d (%.2f%%), Dt=%fs, Avg=%fs, Eff: %f/%f, Obj: %f/%f, Weight: %f'
              % (epoch_num, epochs, epoch_progress * 100, epoch_dts[-1], np.mean(epoch_dts), 
                 test_eff, train_eff,
                 test_objective, train_objective,
                 params_weight))

        if 'fig' in kwargs['kwargs']:
            fig = kwargs['kwargs']['fig']
            plots_mon(fig, nn, epoch_num, test_eff, train_eff)

        return

# diagnosis and training functions

def plots_init(nn, epochs, nw):
    ''' Prepare figure for interactive plotting and return figure object. '''

    fig, _ = plt.subplots(sharex=True, sharey=True, figsize=(10, 10))
    hr = [ 1 ] * (1 + nw + len(nn.layers[1:]))
    hr[0] = 2
    wr = [ 1 ] * (1 + nw)
    wr[0] = .1
    G = gridspec.GridSpec(1 + nw + len(nn.layers[1:]), 1 + nw,
                         width_ratios=wr, height_ratios=hr)
   
    plt.subplot(G[0, :])
    plt.title('Efficiency Monitor')
    plt.axis([0, epochs, .9, 1])
    
    a = 0
    b = 0
    for i in range(min(nn.layers[0].num_out, nw*nw)):
        ax = plt.subplot(G[1 + b, 1 + a])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        a += 1
        if a >= nw:
            a = 0
            b += 1
        plt.title('%d' % i)

    for i in range(len(nn.layers[1:])):
        plt.subplot(G[1 + nw + i, 1:])
        plt.title('W^%d' % (i+1))
    
    ax = plt.subplot(G[1:, 0])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=-1, vmax=1))
    sm._A = []
    plt.colorbar(sm, cax=ax)

    G.tight_layout(fig)
    plt.ion()
    plt.show()

    return fig

def plots_mon(fig, nn, epoch, test_eff, train_eff):
    ''' Draw interactive monitor plots. '''
        
    plt.sca(fig.axes[0])
    plt.scatter(epoch, test_eff, label='efficiency', color='r', s=3)
    plt.scatter(epoch, train_eff, label='training', color='b', s=3)
    # plt.legend(['test', 'training'], loc=4)
        
    W = nn.layers[0].get_params()
    for i in range(nn.layers[0].num_out):
        ax = plt.sca(fig.axes[1+i]) 
        im = plt.imshow(W[i][:28*28].reshape(28,28))

    for i in range(1, len(nn.layers)):
        W = nn.layers[i].get_params()
        plt.sca(fig.axes[nn.layers[0].num_out+i]) 
        plt.imshow(W)

    fig.canvas.draw()
    plt.pause(0.00001)

    return

def plots_end():
    ''' Tidy up after interactive plotting. '''

    plt.ioff()
    plt.show()

    return

def show_error_candidates(nn, data):
    ''' Plot a couple of candidates that were predicted wrongly. '''

    print('Showing some error candidates...')

    _, error_list = nn.test_network(data)
    
    fig = plt.figure('Errors', figsize=(10,10))
    num = 1
    for i, l, p in error_list[:6*6]:
        plt.subplot(6, 6, num)
        num += 1
        plt.imshow(data[i][0].reshape(28,28),cmap='Blues')
        plt.title('#%d: l=%d, p=%d' % (i, l, p))

    fig.tight_layout() 
    plt.show()

    print('Done.')
    
    return

## MAIN ##########################################

data = mnist.MNIST('./MNIST/')
print('%d test images' % len(data.test_imgs))
print('%d training images' % len(data.train_imgs))

# build neural network

# one hidden layer with 36 neurons, learning rate = 3, quadratic loss
# init weights and biases to zero

# reaches >39% in 30 epochs

nn = NN_Diag(data.rows * data.cols, 
             interpreters.ArgMax, 
             objectives.L2)

# square root of number of input neurons
nw = 6

# first layer
nn.add_layer(num=nw * nw, 
             layer=FullyConnected_Diag, 
             activation=activations.Sigmoid,
             train='gradient',
             eta=3, 
             weight=weights.L2, 
             omega=0,
             init="zero"
            )

# output layer
nn.add_layer(num=10, 
             layer=FullyConnected_Diag, 
             activation=activations.Sigmoid,
             train='gradient',
             eta=3, 
             weight=weights.L2, 
             omega=0,
             init="zero"
            )

# prepare data 
train_data = [(data.train_imgs[i].reshape(data.rows * data.cols, 1) / 255.0, \
               data.train_labels[i]) for i in range(len(data.train_imgs))]

test_data = [(data.test_imgs[i].reshape(data.rows * data.cols, 1) / 255.0, \
              data.test_labels[i]) for i in range(len(data.test_imgs))]

# train network
epochs = 30
batch_size = 10
fig = plots_init(nn, epochs, nw)
nn.batch_training(train_data, test_data, epochs, batch_size, fig=fig)
        
# show_error_candidates(nn, test_data)
plots_end()
