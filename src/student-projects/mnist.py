import sys
import struct
from array import array
import numpy as np
   
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class MNIST:
    ''' Data class for MNIST images. '''
    
    def __init__(self, path_name):
        '''
        Read the MNIST data which is assumed in the path `path_name`.
        Optionally one can only load a subset of images by specifying
        `num_test` and `num_train`.
        '''

        try:
            file_test_imgs = open(path_name + 't10k-images-idx3-ubyte', 'rb')
            file_test_labels = open(path_name + 't10k-labels-idx1-ubyte', 'rb')
            file_train_imgs = open(path_name + 'train-images-idx3-ubyte', 'rb')
            file_train_labels = open(path_name + 'train-labels-idx1-ubyte', 'rb')
        except:
            raise 

        self.rows, self.cols, self.test_imgs \
                = self.__read_imgs(file_test_imgs)
        self.test_labels = self.__read_labels(file_test_labels) 

        self.rows, self.cols, self.train_imgs \
                = self.__read_imgs(file_train_imgs)
        self.train_labels = self.__read_labels(file_train_labels) 

        file_test_imgs.close()
        file_test_labels.close()
        file_train_imgs.close()
        file_train_labels.close()
            
        return
    
    def __read_imgs(self, infile):

        magic, size, rows, cols = struct.unpack(">iiii", infile.read(16))
        if magic != 2051:
            raise ValueError('NMIST image file magic is not 2051.')

        raw_data = array("b", infile.read())
           
        images = []
        img_size = rows * cols
        for ptr in range(0, len(raw_data), img_size):
            img = np.array(raw_data[ptr:ptr+img_size], dtype=np.uint8).reshape(rows, cols)
            images.append(img)

        return rows, cols, images
    
    def __read_labels(self, infile):

        magic, size = struct.unpack(">ii", infile.read(8))
        if magic != 2049:
            raise ValueError('NMIST labels file magic is not 2049.') 

        labels = array("b", infile.read())

        return labels

    def show_imgs(self, indices, t='test'):
        '''
        Show a list `indices` of images. If `t=='training'` it will show the
        training images, otherwise the test images.
        '''

        print('Showing %s images: ' % t, end='')

        imgs = self.test_imgs
        labels = self.test_labels
        if t == 'training':
            imgs = self.train_imgs
            labels = self.train_labels

        num = 1
        for i in indices:
            ax = plt.subplot(1, len(indices), num)
            num += 1
            plt.imshow(imgs[i],cmap='Blues')
            print('#%d=%d' % (i, labels[i]), end=', ')

        # flush the output of print() to screen before showing the figure
        sys.stdout.flush()
        plt.show()

        return 
