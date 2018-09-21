import sys
import os
import numpy as np
import errno
import matplotlib.pyplot as plt

key_name_dict = {1:'back', 2:'beak', 3:'belly', 4:'breast', 5:'crown', 6:'forhead', 
        7:'left-eye', 8:'left-leg', 9:'left-wing', 10:'nape', 11:'right-eye', 
        12:'right-leg', 13:'right-wing', 14:'tail', 15:'throat'}

def getPatchId():
    patchId = {}
    for i in xrange(2, 15+1):
        patchId[i - 1] = ((1, i), (key_name_dict[1], key_name_dict[i]))
        #print("%3d -> %3d: %3d: %-10s -> %-10s" % (1, i, i-1, key_name_dict[1], key_name_dict[i]))
    for i in xrange(2, 14+1):
        for j in xrange(i+1, 15+1):
            #key = str(i) + '_' + str(j)
            value = np.arange(14, 16-i-1, -1).sum() + j - i
            patchId[value] = ((i, j), (key_name_dict[i], key_name_dict[j]))
            #print("%2d -> %2d: %3d: %-10s -> %-10s" % (i, j, value, key_name_dict[i], key_name_dict[j]))
    return patchId

def create_if_not_exists(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except (OSError, e):
            if e.errno == errno.EEXIST and os.path.isdir(folder):
                pass

def vis_square(data, title="display"):
    """Take an array of shape (x, height, width) or (x, height, width, 3)
    and visualize each (height, width) thing in a grid of size approx. sqrt(x) by sqrt(x)"""
    # normalize data for display
    data = (data.astype(np.float) - data.min()) / (data.max() - data.min() + 1e-7)
                               
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    
    padding = (((0, n ** 2 - data.shape[0]),
        (0, 1), (0, 1))                 # add some space between filters
            + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data
    #fig = plt.figure()
    #fig.suptitle(title, fontsize=14, fontweight='bold')
    #ax = fig.add_subplot(111)
    #ax.imshow(data)
    #ax.axis('off')
    #plt.show()

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()


