MAML WS 16/17 Handwritten number recognition challenge
======================================================

Task
----

Train neural network to perform classification of handwritten digits from the
MNIST data. The MNIST data comes in two sets, the first one being the training
data, which consists of 60000 images, and the second one being the test data,
which consist of 10000 images. The ranking will be done w.r.t. error rate on
the test data when trained using the training data only.

Rules
-----

1. The network may not be trained with the test data but only with the training
   data or the data derived from the training data.

2. In order to be able to compare the results, let us agree that the network
   must be a dense network, i.e., each neuron of one layer is connected to each
   neuron of the next layer.

3. The network breadth (neurons per layer) and depth (number of layers) is not
   restricted.

4. Training method is not restricted as long as it is only based on the
   training data and does not make use of the test data. Otherwise, you are
   free to choose loss functions, activation functions, regularizations, and
   other tuning techniques to optimize the training.

5. The training should not take more than 15min on my machine ;)


Hints
-----

I prepare a couple of scripts to get you started:

* `download.py` is a script that downloads and decompresses the MNIST data.

* `mist_browser.py` is a script that employs the methods of the module
  `mnist.py` which reads the MNIST data from the local path and displays
  a couple of the images.

* In the folder `../Multilayer Network` you find an implementation of the
  multilayer network that we discussed in one of the discussion sessions.
  Several loss and activation functions are implemented already. The first step
  would be to replace the loading function of the IRIS data with one that loads
  the MNIST data.

* Then the tuning begins:

  * As a first trial use: 28x28=784 input neurons, a second layer of 30 neurons,
    and an output layer of 10 neurons representing the probabilities for the 10
    labels '0, 1, 2, ..., 9'. As loss function use `objectives.L2` and as
    activation functions `actvations.Sigmoid` with a learning rate of
    :math:`\eta=3`. Choose 30 epochs with a mini-batch size of 10 for the
    training which should result in a neural network that reaches 94-96%
    accuracy.

  * As a next step try different architectures, i.e., add a layer with
    a different breadth, try the cross-entropy loss function, and
    regularization techniques such as an L2 norm off the weight matrix
    elements.

  * Finally you could also experiment with expanding your training data set by
    addition rotations, scalings, distortions, etc.

Results
=======

I will post the results and the code of the participants here.
