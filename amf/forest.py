# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: GPL 3.0

import numpy as np
from numba import jitclass
from numba import types
from numba.types import float32, boolean, uint32, string

from .checks import check_X_y, check_array
from .sample import SamplesCollection
from .tree import TreeClassifier
from .tree_methods import tree_partial_fit, tree_predict

spec = [
    ("n_classes", uint32),
    ("n_features", uint32),
    ("n_estimators", uint32),
    ("step", float32),
    ("loss", string),
    ("use_aggregation", boolean),
    ("dirichlet", float32),
    ("split_pure", boolean),
    ("n_jobs", uint32),
    ("random_state", uint32),
    ("verbose", boolean),
    ("trees", types.List(TreeClassifier.class_type.instance_type, reflected=True)),
    ("samples", SamplesCollection.class_type.instance_type),
    ("iteration", uint32),
]


# TODO: it's a mess, many things to check (check_array) and many things should
#  be properties in OnlineForestClassifier

# TODO: we can force pre-compilation when creating the nopython forest


@jitclass(spec)
class AMFClassifierNoPython(object):
    def __init__(
        self,
        n_classes,
        n_features,
        n_estimators,
        step,
        loss,
        use_aggregation,
        dirichlet,
        split_pure,
        n_jobs,
        random_state,
        verbose,
    ):

        self.n_features = n_features
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation
        self.dirichlet = dirichlet
        self.split_pure = split_pure
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.iteration = 0

        samples = SamplesCollection()
        self.samples = samples

        # TODO: reflected lists will be replace by typed list soon...
        trees = [
            TreeClassifier(n_features, n_classes, samples) for _ in range(n_estimators)
        ]
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
        scores.fill(0.0)
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
            raise RuntimeError("You must call ``partial_fit`` before ``predict``.")


class AMFClassifier(object):
    """Aggregated Mondrian Forest classifier for online learning. This algorithm
    is truly online, in the sense that a single pass is performed, and one can
    ask for predictions anytime.

    Parameters
    ----------
    n_classes : `int`
        Number of excepted classes in the labels. This is required since we
        don't know the number of classes in advance in a online setting.

    n_estimators : `int`, optional (default=10)
        Number of trees to grow in the forest.

    step : `float`, optional (default=1)
        Step-size for the aggregation weights. Default is 1 for classification,
        which is typically the best choice.

    loss : 'str', optional (default='log')
        The loss used for the computation of the aggregation weights. Only `log`
        is supported for now, namely the log-loss for multi-class
        classification.

    use_aggregation : `bool`, optional (default=True)
        Whether to use aggregation in each tree. It is highly recommended to
        leave it as `True`.

    dirichlet : `float` or `None`, optional (default=None)
        Each node in a tree predicts according to the distribution of the labels
        it contains. This distribution is regularized using a "Jeffreys" prior
        with parameter `dirichlet`. For each class with `count` labels in the
        node and `n_samples` samples in it, the prediction of a node is given by
        (count + dirichlet) / (n_samples + dirichlet * n_classes).

        Default is dirichlet=0.5 for n_classes=2 and dirichlet=0.01 otherwise.

    split_pure : `bool`, optional (default=False)
        Whether we split nodes that contain only one class of labels. Default is
        False.

    n_jobs : `int`, optional (default=1)
        The number of threads used to grow the tree in parallel. This has no
        effect for now, the default is n_jobs=1, namely single-threaded.

    random_state : `int` or `None`, optional (default=None)
        Controls the randomness involved in the trees growing, which is a highly
        randomized process.

    verbose : `bool`, optional (default=True)
        Whether a bar showing progress for `partial_fit`, `predict_proba` and
        `predict` should be used.

    Attributes
    ----------
    n_classes = `int`
        Number of excepted classes in the labels.

    n_features : `int`
        The number of features from the training dataset (passed to ``fit``)

    # TODO: add missing attributes

    Notes
    -----
    Note that all the parameters of AMFClassifier become read-only after the
    first call to `partial_fit`

    References
    ----------
    TODO : add the reference of the paper
    """

    def __init__(
        self,
        n_classes,
        n_estimators=10,
        step=1.0,
        loss="log",
        use_aggregation=True,
        dirichlet=None,
        split_pure=False,
        n_jobs=1,
        random_state=None,
        verbose: bool = True,
    ):

        # We will instantiate the numba class when data is passed to
        # `partial_fit`, since we need to know about `n_features` among others
        self.no_python = None

        self.n_classes = n_classes
        self._n_features = None
        self.n_estimators = n_estimators
        self.step = step
        self.loss = loss
        self.use_aggregation = use_aggregation

        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self.split_pure = split_pure
        self.n_jobs = n_jobs

        # TODO: deal with random_state
        self.random_state = 0
        self.verbose = verbose

    def partial_fit(self, X, y):
        n_samples, n_features = X.shape
        # First,ensure that X and y are C-contiguous and with float32 dtype
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="float32",
            order="C",
            copy=False,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            multi_output=False,
            ensure_min_samples=1,
            ensure_min_features=1,
            y_numeric=True,
            estimator="OnlineForestClassifier",
        )
        # TODO: test that y.min and y.max is in [0, n_classes-1]s
        # TODO: raise a warning is a label is not present ? This could be
        #  optional

        # This is the first call to `partial_fit`, so we need to instantiate
        # the no python class
        if self.no_python is None:
            self._n_features = n_features
            # max_nodes_with_memory_in_tree = \
            #     int(1024 ** 2 * self.memory
            #         / (8 * self.n_estimators * n_features))
            self.no_python = AMFClassifierNoPython(
                self.n_classes,
                self.n_features,
                self.n_estimators,
                self.step,
                self.loss,
                self.use_aggregation,
                self.dirichlet,
                self.split_pure,
                self.n_jobs,
                self.random_state,
                self.verbose,
            )
        else:
            pass
            # TODO: test that y.min and y.max is in [0, n_classes-1]
            # TODO: test that n_features is unchanged
            # TODO: raise a warning is a label is not present ? This could be
            #  optional

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

        X = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=["float32"],
            order="C",
            copy=False,
            force_all_finite=True,
            ensure_2d=True,
            allow_nd=False,
            ensure_min_samples=1,
            ensure_min_features=1,
            estimator="OnlineForestClassifier",
        )

        scores = np.empty((X.shape[0], self.n_classes), dtype="float32")
        if not self.no_python:
            raise RuntimeError("You must call ``partial_fit`` before")
        else:
            pass
            # TODO: check sur X, check n_features
            # TODO: X = safe_array(X, dtype='float32')
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

        columns = [
            "id",
            "parent",
            "left",
            "right",
            "depth",
            "is_leaf",
            "feature",
            "threshold",
            "time",
            "n_samples",
            "memory_range_min",
            "memory_range_max",
            "counts",
        ]

        data = {
            "id": index,
            "parent": parent,
            "left": left,
            "right": right,
            "depth": depth,
            "feature": feature,
            "threshold": threshold,
            "is_leaf": is_leaf,
            "time": time,
            "n_samples": n_samples,
            "memory_range_min": [tuple(t) for t in memory_range_min],
            "memory_range_max": [tuple(t) for t in memory_range_max],
            "counts": [tuple(t) for t in counts],
        }
        df = pd.DataFrame(data, columns=columns)
        return df

    # TODO: code properties for everything
    @property
    def n_classes(self):
        return self._n_classes

    @n_classes.setter
    def n_classes(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_classes` after calling `partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_classes` must be of type `int`")
            elif val < 2:
                raise ValueError("`n_classes` must be >= 2")
            else:
                self._n_classes = val

    @property
    def n_features(self):
        return self._n_classes

    @n_features.setter
    def n_features(self, val):
        raise ValueError("`n_features` is a readonly attribute")

    @property
    def n_estimators(self):
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_estimators` after calling " "`partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_estimators` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_estimators` must be >= 1")
            else:
                self._n_estimators = val

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `n_jobs` after calling " "`partial_fit`"
            )
        else:
            if not isinstance(val, int):
                raise ValueError("`n_jobs` must be of type `int`")
            elif val < 1:
                raise ValueError("`n_jobs` must be >= 1")
            else:
                self._n_jobs = val

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, val):
        if self.no_python:
            raise ValueError("You cannot modify `step` after calling " "`partial_fit`")
        else:
            if not isinstance(val, float):
                raise ValueError("`step` must be of type `float`")
            elif val <= 0:
                raise ValueError("`step` must be > 0")
            else:
                self._step = val

    @property
    def use_aggregation(self):
        return self._use_aggregation

    @use_aggregation.setter
    def use_aggregation(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `use_aggregation` after " "calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`use_aggregation` must be of type `bool`")
            else:
                self._use_aggregation = val

    @property
    def split_pure(self):
        return self._split_pure

    @split_pure.setter
    def split_pure(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `split_pure` after " "calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`split_pure` must be of type `bool`")
            else:
                self._split_pure = val

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        if self.no_python:
            raise ValueError(
                "You cannot modify `verbose` after " "calling `partial_fit`"
            )
        else:
            if not isinstance(val, bool):
                raise ValueError("`verbose` must be of type `bool`")
            else:
                self._verbose = val

    @property
    def loss(self):
        return "log"

    @loss.setter
    def loss(self, val):
        pass

    def __repr__(self):
        r = "AMFClassifier"
        r += "(n_classes={n_classes}, ".format(n_classes=self.n_classes)
        r += "n_estimators={n_estimators}, ".format(n_estimators=self.n_estimators)
        r += "step={step}, ".format(step=self.step)
        r += "loss={loss}, ".format(loss=self.loss)
        r += "use_aggregation={use_aggregation}, ".format(
            use_aggregation=self.use_aggregation
        )
        r += "dirichlet={dirichlet}, ".format(dirichlet=self.dirichlet)
        r += "split_pure={split_pure}, ".format(split_pure=self.split_pure)
        r += "n_jobs={n_jobs}, ".format(n_jobs=self.n_jobs)
        r += "random_state={random_state}, ".format(random_state=self.random_state)
        r += "verbose={verbose})".format(verbose=self.verbose)
        return r

    # TODO: properties for dirichlet
    # TODO: properties for random_state
