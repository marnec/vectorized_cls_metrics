import numpy as np
from scipy.sparse import coo_matrix


def confusion_matrix(y_true, y_pred):
    sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    return coo_matrix((sample_weight, (y_true, y_pred))).toarray()


if __name__ == "__main__":
    # tp: 1, tn: 2, fn: 3, fp: 4
    ytrue = np.array([1, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    ypred = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    confusion_matrix(ytrue, ypred)
