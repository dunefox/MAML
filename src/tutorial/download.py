import urllib.request

# download MNIST data

mnist_url_path = "http://yann.lecun.com/exdb/mnist/"
mnist_files = ( "train-images-idx3-ubyte.gz",
               "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz",
               "t10k-labels-idx1-ubyte.gz"  )
mnist_dest = "./MNIST"

for f in mnist_files:
    urllib.request.urlretrieve(mnist_url_path + f, f)
