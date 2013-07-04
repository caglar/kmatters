import os, logging
_logger = logging.getLogger(__name__)

import cPickle as pkl
import numpy as np
N = np
from pylearn2.datasets import dense_design_matrix

class Pentomino(dense_design_matrix.DenseDesignMatrix):

    def __init__(self,
            which_set,
            start=None,
            stop=None,
            dir=None,
            use_binary=True,
            one_hot=True,
            preprocessor=None,
            names=None):

        # note: there is no such thing as the cifar10 validation set;
        # quit pretending that there is.
        self.args = locals()

        assert which_set in ["valid", "test", "train"]
        # we define here:
        self.dtype  = 'float32'
        ntrain = 40000
        nvalid = 0  # artifact, we won't use it
        ntest  = 20000
        self.n_multi_classes = 10
        self.which_set = which_set
        if type(names) is not list:
            names = [names]

        if names is None:
            raise ValueError("Dataset names shouldn't be empty. Choose one of the following: valid, test, train")

        if dir is None:
            dir = "/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/"

        self.img_shape = (64, 64)
        self.img_size = np.prod(self.img_shape)
        self.use_binary = use_binary
        X, y = self._load_data(dir, names)

        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.img_size

        X = X[start:stop]
        y = y[start:stop]

        if self.which_set == "test":
            assert X.shape[0] == 20000

        if self.use_binary:
            labels = self.binarize_labels(y)
            y = np.asarray(labels, dtype="uint8")
            y = y.reshape((y.shape[0], 1))

        if one_hot:
            one_hot = np.zeros((y.shape[0],2),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,y[i]] = 1.
            y = one_hot

        view_converter = dense_design_matrix.DefaultViewConverter((64, 64, 1))
        super(Pentomino, self).__init__(X=X, y=y, view_converter=view_converter)
        if preprocessor:
            preprocessor.apply(self)

    def _load_data(self, dir, names):
        X = None
        y = None
        for name in names:
            if name.endswith("npy"):
                data = np.load(dir + name)
                if X is None:
                    X = data[0]
                else:
                    X = np.vstack((X, data[0]))
                if y is None:
                    y = data[1]
                else:
                    y = np.vstack((y, data[1]))

            else:
                data = pkl.load(open(dir + name, "rb"))
                if X is None:
                    X = data[0]
                else:
                    X = np.vstack((X, data[0]))
                if y is None:
                    y = data[1]
                else:
                    y = np.vstack((y, data[1]))
        return (np.asarray(X.tolist(), dtype=self.dtype), np.asarray(y.tolist(), dtype="uint8"))

    def binarize_labels(self, labels=None):
        """
        Convert the labels into the binary format for the second phase task.
        """
        #Last label is for the images without different object.
        last_lbl = self.n_multi_classes
        binarized_lbls = []
        for label in labels:
            if label == last_lbl:
                binarized_lbls.append(0)
            else:
                binarized_lbls.append(1)
        return binarized_lbls

"""Test 1
pento = Pentomino(which_set="train", start=100, stop=10000,
dir="/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/",
names=["pento64x64_40k_seed_2162026_64patches.npy"])
"""

"""Test 2
pento = Pentomino(which_set="train", start=100, stop=10000,
dir="/RQexec/gulcehre/datasets/pentomino/pento_64x64_8x8patches/",
names="pento64x64_40k_seed_2162026_64patches.npy")
"""

