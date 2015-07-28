__author__ = 'matteo'

# import
from csv import reader
import numpy as np
from sklearn import decomposition
from sklearn.lda import LDA

#load data
def load_my_data(path):

    # initialize
    X = []
    y = []

    # load data
    with open(path) as csv:

        r = reader(csv)
        r.next()

        for line in r:
            X.append(map(float, line[0:len(line)-1]))
            y.append(int(line[-1]))

    X = np.asarray(X)
    y = np.asarray(y)
    return (X,y)


def feat_extraction(X,y,D):

    # usupervised feature extraction: Principal Component Analysis
    pca = decomposition.PCA(n_components=D)
    pca.fit(X)
    X_pca = pca.transform(X)

    # supervised feature extraction: Linear Discriminative Analysis
    lda = LDA(n_components=D)
    lda.fit(X,y)
    X_lda = lda.transform(X)

    return (X_pca,X_lda)