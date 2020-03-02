# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: GPL 3.0

import numpy as np
from numba import jitclass
from numba import types
from numba.types import float32, boolean, uint32, int32, string

from .sample import SamplesCollection
from .tree import TreeClassifier
from .tree_methods import tree_partial_fit, tree_predict

spec = [
    ('n_features', uint32),
    ('n_classes', uint32),
    ('n_estimators', uint32),
    ('step', float32),
    ('criterion', string),
    ('use_aggregation', boolean),
    ('dirichlet', float32),
    ('split_pure', boolean),
    ('min_samples_split', int32),
    ('max_features', uint32),
    ('n_threads', uint32),
    ('seed', uint32),
    ('verbose', boolean),
    ('print_every', uint32),
    ('max_nodes_with_memory_in_tree', uint32),
    ('trees', types.List(TreeClassifier.class_type.instance_type,
                         reflected=True)),
    ('samples', SamplesCollection.class_type.instance_type),
    ('iteration', uint32)
]


# TODO: it's a mess, many things to check (check_array) and many things should
#  be properties in OnlineForestClassifier

# TODO: we can force pre-compilation when creating the nopython forest


@jitclass(spec)
class OnlineForestClassifierNoPython(object):

    def __init__(self, n_features, n_classes, n_estimators, step, criterion,
                 use_aggregation, dirichlet, split_pure, min_samples_split,
                 max_features, n_threads, seed, verbose, print_every,
                 max_nodes_with_memory_in_tree):

        self.n_features = n_features
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.step = step
        self.criterion = criterion
        self.use_aggregation = use_aggregation
        self.dirichlet = dirichlet
        self.split_pure = split_pure
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.print_every = print_every
        self.max_nodes_with_memory_in_tree = max_nodes_with_memory_in_tree

        self.iteration = 0

        samples = SamplesCollection()
        self.samples = samples

        # TODO: reflected lists will be replace by typed list soon...
        trees = [TreeClassifier(n_features, n_classes, samples) for _ in
                 range(n_estimators)]
        self.trees = trees

    def partial_fit(self, X, y):
        n_samples_batch, n_features = X.shape

        # We need at least the actual number of nodes + twice the extra samples

        # Let's reserve nodes in advance
        # print('Reserve nodes in advance')
        for tree in self.trees:
            n_nodes_reserved = tree.nodes.n_nodes + 2 * n_samples_batch
            # print('n_nodes_reserved:', n_nodes_reserved)
            tree.nodes.reserve_nodes(n_nodes_reserved)
        # print('Done.')

        # First, we save the new batch of data
        # print('Append the new batch of data')
        n_samples_before = self.samples.n_samples
        self.samples.append(X, y)
        # print('Done.')

        for i in range(n_samples_before, n_samples_before + n_samples_batch):
            # Then we fit all the trees using all new samples
            for tree in self.trees:
                # print('i:', i)
                tree_partial_fit(tree, i)

            self.iteration += 1

    def predict(self, X, scores):
        scores.fill(0.)
        n_samples_batch, _ = X.shape
        if self.iteration > 0:
            scores_tree = np.empty(self.n_classes, float32)
            for i in range(n_samples_batch):
                # print('i:', i)
                scores_i = scores[i]
                x_i = X[i]
                # print('x_i:', x_i)
                # The prediction is simply the average of the predictions
                for tree in self.trees:
                    tree_predict(tree, x_i, scores_tree, self.use_aggregation)
                    # print('scores_tree:', scores_tree)
                    scores_i += scores_tree
                scores_i /= self.n_estimators
                # print('scores_i:', scores_i)
        else:
            raise RuntimeError(
                "You must call ``partial_fit`` before ``predict``.")


class OnlineForestClassifier(object):

    def __init__(self, n_classes, n_estimators: int = 10, step: float = 1.,
                 criterion: str = 'log', use_aggregation: bool = True,
                 dirichlet: float = None, split_pure: bool = False,
                 min_samples_split: int = -1, max_features: int = -1,
                 n_threads: int = 1, seed: int = -1,
                 verbose: bool = True, print_every=1000, memory: int = 512):

        # TODO: many attributes can't be changed once the no python object is created
        self.no_python = None
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.step = step
        self.criterion = criterion
        self.use_aggregation = use_aggregation

        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self.split_pure = split_pure
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.print_every = print_every
        self.memory = memory

        self.n_features = None
        self._fitted = False

    def partial_fit(self, X, y):
        # X = safe_array(X, dtype='float32')
        # y = safe_array(y, dtype='float32')
        n_samples, n_features = X.shape

        if self.no_python is None:
            self.n_features = n_features
            max_nodes_with_memory_in_tree = \
                int(1024 ** 2 * self.memory
                    / (8 * self.n_estimators * n_features))

            self.no_python = OnlineForestClassifierNoPython(
                n_features,
                self.n_classes,
                self.n_estimators,
                self.step,
                self.criterion,
                self.use_aggregation,
                self.dirichlet,
                self.split_pure,
                self.min_samples_split,
                self.max_features,
                self.n_threads,
                self.seed,
                self.verbose,
                self.print_every,
                max_nodes_with_memory_in_tree
            )
            self._fitted = True

        self.no_python.partial_fit(X, y)
        return self

    def predict_proba(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix to predict for.

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, n_classes)
            Returns predicted values.
        """
        import numpy as np

        scores = np.empty((X.shape[0], self.n_classes), dtype='float32')
        if not self._fitted:
            raise RuntimeError("You must call ``fit`` before")
        else:
            pass
            # X = safe_array(X, dtype='float32')
        self.no_python.predict(X, scores)
        return scores

    # def predict(self, X):
    #     if not self._fitted:
    #         raise RuntimeError("You must call ``fit`` before")
    #     else:
    #         X = safe_array(X, dtype='float32')
    #         scores = self.predict_proba(X)
    #         return scores.argmax(axis=1)

    def get_nodes_df(self, idx_tree):
        import pandas as pd

        tree = self.no_python.trees[idx_tree]
        nodes = tree.nodes
        n_nodes = nodes.n_nodes

        index = nodes.index[:n_nodes]
        parent = nodes.parent[:n_nodes]
        left = nodes.left[:n_nodes]
        right = nodes.right[:n_nodes]
        feature = nodes.feature[:n_nodes]
        threshold = nodes.threshold[:n_nodes]
        time = nodes.time[:n_nodes]
        depth = nodes.depth[:n_nodes]
        memory_range_min = nodes.memory_range_min[:n_nodes]
        memory_range_max = nodes.memory_range_max[:n_nodes]
        n_samples = nodes.n_samples[:n_nodes]
        # weight = nodes.weight[:n_nodes]
        # log_weight_tree = nodes.log_weight_tree[:n_nodes]
        is_leaf = nodes.is_leaf[:n_nodes]
        # is_memorized = nodes.is_memorized[:n_nodes]
        counts = nodes.counts[:n_nodes]

        columns = ['id', 'parent', 'left', 'right', 'depth', 'is_leaf',
                   'feature', 'threshold', 'time', 'n_samples',
                   'memory_range_min', 'memory_range_max', 'counts']

        data = {
            'id': index, 'parent': parent, 'left': left, 'right': right,
            'depth': depth, 'feature': feature, 'threshold': threshold,
            'is_leaf': is_leaf, 'time': time, 'n_samples': n_samples,
            'memory_range_min': [tuple(t) for t in memory_range_min],
            'memory_range_max': [tuple(t) for t in memory_range_max],
            'counts': [tuple(t) for t in counts]
        }
        df = pd.DataFrame(data, columns=columns)
        return df
