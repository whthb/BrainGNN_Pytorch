from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train_val_test_split(kfold=5, fold=0, labels=None, n_sub=None):
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        n_sub = len(labels)
    elif n_sub is None:
        n_sub = 1035

    indices = np.arange(n_sub)

    if labels is None:
        kf = KFold(n_splits=kfold, random_state=123, shuffle=True)
        kf2 = KFold(n_splits=kfold - 1, shuffle=True, random_state=666)
    else:
        kf = StratifiedKFold(n_splits=kfold, random_state=123, shuffle=True)
        kf2 = StratifiedKFold(n_splits=kfold - 1, shuffle=True, random_state=666)


    test_index = list()
    train_index = list()
    val_index = list()

    split_iter = kf.split(indices, labels) if labels is not None else kf.split(indices)
    for tr, te in split_iter:
        test_index.append(te)
        if labels is None:
            tr_id, val_id = list(kf2.split(tr))[0]
        else:
            tr_id, val_id = list(kf2.split(tr, labels[tr]))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id
