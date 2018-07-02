import numpy as np

# for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# neural network classes
import multilayernet
import objectives
import interpreters
import activations
import layers
import db

# add some function sto the classes that help with the analysis

class NN_Diag(multilayernet.NN):
    ''' Neural network with diagnosis functions. '''

    def epoch_monitor(self, **kwargs):
        ''' plot results after training of each epoch '''

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

        epoch_num = len(epoch_dts)
        epoch_progress = float(epoch_num) / epochs

        print('> Epoch=%d/%d (%.2f%%), Dt=%fs, Avg=%fs, Eff: %f/%f, Obj: %f/%f'
              % (epoch_num, epochs, epoch_progress * 100, epoch_dts[-1],
                 np.mean(epoch_dts), test_eff, train_eff, test_objective,
                 train_objective))

        if 'fig' in kwargs['kwargs']:
            fig = kwargs['kwargs']['fig']
            self.plots_mon(epoch_num, epochs, train_eff, test_eff, test_data, train_data, 3, fig)

        return

    def plots_mon(self, epoch_num, epochs, train_eff, test_eff,  test_data, train_data, n_out, fig):
        ''' Draw interactive monitor plots. '''

        plot_eff(epoch_num, epochs, train_eff, test_eff, fig)
        plot_decision_regions(self, test_data, train_data, self.n[-1], 0.1, fig)
        fig.canvas.draw()
        plt.show()
        # matplotlib bug workaround
        plt.pause(0.00001)

        return

# diagnosis and training functions

def plots_init(nn, train_data, test_data):
    ''' Prepare figure for interactive plotting and return figure object. '''

    fig, axes = plt.subplots(figsize=(10, 20))
    plt.subplot(2,1,1)
    plt.title('Efficiency Monitor')
    plt.subplot(2,1,2)
    plot_decision_regions(nn, train_data, test_data, 3, 0.1, fig)
    plt.ion()
    plt.show()

    return fig

def plots_end():
    ''' Tidy up after interactive plotting. '''

    plt.ioff()
    plt.show()

    return

def plot_data(X, Y, n_classes, m, fig):
    ''' plot X, Y data '''

    # color map
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_classes))

    # plot data with colors according to class labels
    for l, c in zip(range(n_classes), colors):
        xs = []
        for xi, yi in zip(X, Y):
            if yi == l:
                xs.append(xi)
        xs = np.array(xs)
        plt.scatter(xs[:,0], xs[:,1], color=c, marker=m, edgecolor='black')

    return

def plot_eff(epoch_num, epochs, train_eff, test_eff, fig):
    ''' plot the efficiency '''

    plt.figure(fig.number)
    axs = fig.axes[0]

    axs.set_xlim(0, epochs)
    axs.set_ylim(0, 1)

    axs.scatter(epoch_num, test_eff, label='efficiency', color='r', s=3)
    axs.scatter(epoch_num, train_eff, label='training', color='b', s=3)
    axs.legend(['test', 'training'], loc=4)

    return

def plot_decision_regions(nn, train, test, n_classes, resolution, fig):
    ''' plot the decission boundary '''

    X = np.array( [ test[i][0] for i in range(len(test)) ] )
    Y = np.array( [ test[i][1] for i in range(len(test)) ] )

    X_train = np.array( [ train[i][0] for i in range(len(train)) ] )
    Y_train = np.array( [ train[i][1] for i in range(len(train)) ] )

    # set up a 2d mesh of data points with resolution `resolution`
    x1_min, x1_max = X[:,0].min() - 2, X[:,0].max() + 2
    x2_min, x2_max = X[:,1].min() - 2, X[:,1].max() + 2

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # start new plot
    plt.figure(fig.number)
    axs = fig.axes[1]
    plt.cla()

    # make fictitious feature data out of the above 2d mesh
    Z = []
    for x1, x2 in zip(xx1.ravel(), xx2.ravel()):
        xin = np.array( [[x1],[x2]] )
        xout, p = nn.feed_forward(xin)
        Z.append(p)
    Z = np.array(Z).reshape(xx1.shape)

    # plot the mesh as contour plot
    axs.contourf(xx1, xx2, Z, alpha=0.4)
    axs.set_xlim(xx1.min(), xx1.max())
    axs.set_ylim(xx2.min(), xx2.max())

    # plot training data with 'x's
    plot_data(X_train, Y_train, n_classes, 'x', fig)
    # plot unknown data with 'o's
    plot_data(X, Y, n_classes, 'o', fig)

    return

## MAIN ##########################################

data = db.IRIS('./IRIS/iris.csv')
print('%d test samples' % len(data.test_features))
print('%d training samples' % len(data.train_features))

# build neural network

n_in = data.dim_features
n_out = 3

print('Input dimension:', n_in)
print('Output dimension:', n_out)

nn = NN_Diag(n_in, interpreters.ArgMax, objectives.L2)
nn.add_layer(num=50, layer=layers.FullyConnected,
             activation=activations.Sigmoid, eta=.2)
nn.add_layer(num=20, layer=layers.FullyConnected,
             activation=activations.Sigmoid, eta=.2)
nn.add_layer(num=n_out, layer=layers.FullyConnected,
             activation=activations.Sigmoid, eta=.4)

# prepare data
train_data = [(data.train_features[i].reshape(data.dim_features, 1), data.train_labels[i])
              for i in range(len(data.train_features))]

test_data = [(data.test_features[i].reshape(data.dim_features, 1), data.test_labels[i])
             for i in range(len(data.test_features))]

epochs = 400
batch_size = 50
nn.batch_training(train_data, test_data, epochs, batch_size)

fig = plots_init(nn, train_data, test_data)
epochs = 400
batch_size = 50
nn.batch_training(train_data, test_data, epochs, batch_size, fig=fig)
plots_end()
