import numpy as np

# for plotting
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mnist
from sklearn import svm

## MAIN ##########################################

data = mnist.MNIST('./MNIST/')
print('%d test images' % len(data.test_imgs))
print('%d training images' % len(data.train_imgs))

# build neural network

train_len = int(len(data.train_imgs)/1)
test_len = int(len(data.test_imgs)/1)

# prepare data 
train_imgs = np.array([data.train_imgs[i].reshape(data.rows * data.cols) / 255.0 for i in range(train_len)])
train_labels = np.array([data.train_labels[i] for i in range(train_len)])

test_imgs = np.array([data.test_imgs[i].reshape(data.rows * data.cols) / 255.0 for i in range(test_len)])
test_labels = np.array([data.test_labels[i] for i in range(test_len)])

# train
model = svm.SVC(gamma=0.001)
model.fit(train_imgs, train_labels)

# test
predictions = [int(a) for a in model.predict(test_imgs)]
num_correct = sum(int(a == y) for a, y in zip(predictions, test_labels))

print("Support Vector Machine Efficiency: %03f" % (float(num_correct) / len(test_labels)))
