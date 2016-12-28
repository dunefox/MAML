import numpy as np

# for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# neural network classes
import mnist

## MAIN ##########################################

data = mnist.MNIST('./')
print('%d test images' % len(data.test_imgs))
print('%d training images' % len(data.train_imgs))

data.show_imgs([1, 2, 3], t='test')
