import urllib.request
import gzip

# download MNIST data

mnist_url_path = "http://yann.lecun.com/exdb/mnist/"

mnist_files_gz = ( "train-images-idx3-ubyte.gz",
               "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz",
               "t10k-labels-idx1-ubyte.gz"  )

mnist_files = ( "train-images-idx3-ubyte",
               "train-labels-idx1-ubyte",
               "t10k-images-idx3-ubyte",
               "t10k-labels-idx1-ubyte"  )

for f, g in zip(mnist_files_gz, mnist_files):

    urllib.request.urlretrieve(mnist_url_path + f, f)

    inf = gzip.GzipFile(f, 'rb')
    indata = inf.read()
    inf.close()
    
    outf = open(g, 'wb')
    outf.write(indata)
    outf.close()
