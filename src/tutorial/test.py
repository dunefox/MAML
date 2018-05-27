import mnist

data = mnist.MNIST('MNIST/')
print('%d test images' % len(data.test_imgs))
print('%d training images' % len(data.train_imgs))
