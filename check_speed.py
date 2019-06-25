
import numpy as np
from numpy.random import uniform
from numba.types import float32, boolean, uint32, uint8, int32, string, \
    void, Tuple


from random import expovariate

from math import log, exp
from numba import types, typed
from numba import jitclass, njit

from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from amf import OnlineForestClassifier


np.set_printoptions(precision=3)


n_samples = 5000
n_features = 100
n_classes = 2
# max_iter = 40


X, y = make_blobs(n_samples=n_samples, n_features=n_features, cluster_std=3.,
                  centers=n_classes, random_state=123)


# X = MinMaxScaler().fit_transform(X)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# Precompilation
of = OnlineForestClassifier(n_classes=n_classes, seed=1234,
                            use_aggregation=True,
                            n_estimators=1, split_pure=True,
                            dirichlet=0.5, step=1.)
of.partial_fit(X_train[:5], y_train[:5])
of.predict_proba(X_test[:5])


repeats = 5
for repeat in range(1, repeats + 1):
    print('-' * 16)
    of = OnlineForestClassifier(n_classes=n_classes, seed=1234,
                                use_aggregation=True,
                                n_estimators=20, split_pure=True,
                                dirichlet=0.5, step=1.)
    t1 = time()
    of.partial_fit(X_train, y_train)
    t2 = time()
    print('time fit % d:' % repeat, t2 - t1, 'seconds')

    t1 = time()
    y_pred = of.predict_proba(X_test)
    t2 = time()
    print('time predict %d:' % repeat, t2 - t1, 'seconds')
    roc_auc = roc_auc_score(y_test, y_pred[:, 1])
    print('ROC AUC: %.2f' % roc_auc)
