
# // TODO: change the Dirichlet parameter
# // TODO: feature_importances with a nullptr by default
# // TODO: subsample parameter, default 0.5
# // TODO: tree aggregation
# // TODO: memory optimization (a FeatureSplitter), maximum (sizeof(uint8_t)
# // splits)), a set of current splits
# // TODO: only binary features version ?
# // TODO: add a min_sample_split option
# // TODO: column subsampling and feature selection
# // TODO: class balancing: sample_weights or class_weight option
# // TODOs 2019 / 02 / 26
# // TODO: a tree must know how to add_node_memorized_range() a tree must now the latest node with memory
# // TODO: a tree must have a max_nodes_memorized_range and a n_nodes_memorized_range
# // TODO: range computation std::pair<float, float> NodeClassifier::range(uint32_t j) const should be performed by collecting all the sample in the leaves....
# // TODO: In node, only use these functions in copy constructor
# // TODO: add_node only creates nodes with no memory
# // TODO: _n_nodes_with_memory can increase and decrease. Don't these for n_samples in update_sample_type, test the memory remaining instead
# // TODO: use std::multimap to map n_samples to node_index and std::map to map node_index to n_samples. A node must always increment this itself
# // TODO: this would solve the problem with std::map with a crazy order
# // TODO: is_dirac verifier que si le label a la meme couleur que le noeud, alors on ne split pas, sinon on split


# A faire pour la suite

# 1. remplacer multinomial par autre chose

import numpy as np
from numpy.random import uniform
from numba.types import float32, boolean, uint32, uint8, int32, string, \
    void, Tuple


from random import expovariate

from math import log, exp
from numba import types, typed
from numba import jitclass, njit

from time import time

np.set_printoptions(precision=4)


# TODO: c'est pourri comme strategie si on est vraiment online... faudrait forcer les batch, attendre d'avoir assez de donnees avec de refaire les tableaux





# void TreeClassifier::predict(const ArrayFloat &x_t, ArrayFloat &scores, bool use_aggregation) {
#   uint32_t leaf = get_leaf(x_t);
#   if (!use_aggregation) {
#     node(leaf).predict(scores);
#     return;
#   }
#   uint32_t current = leaf;
#   // The child of the current node that does not contain the data
#   ArrayFloat pred_new(n_classes());
#   while (true) {
#     NodeClassifier &current_node = node(current);
#     if (current_node.is_leaf()) {
#       current_node.predict(scores);
#     } else {
#       float w = std::exp(current_node.weight() - current_node.weight_tree());
#       // Get the predictions of the current node
#       current_node.predict(pred_new);
#       for (uint8_t c = 0; c < n_classes(); ++c) {
#         scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c];
#       }
#     }
#     // Root must be updated as well
#     if (current == 0) {
#       break;
#     }
#     current = current_node.parent();
#   }
# }



def bidouille():

    # from tick.online import OnlineForestClassifier
    n_features = 10
    n_classes = 2
    n_nodes_reserved = 4
    samples = SamplesCollection()
    tree = TreeClassifier(n_features, n_classes, samples)
    tree.reserve_nodes(n_nodes_reserved)

    # nodes = NodeCollection(n_features, n_classes, n_nodes_reserved)
    nodes = tree.nodes
    nodes.add_node(0, 1e-2)
    nodes.add_node(0, 1e-2)
    nodes.add_node(1, 3e-2)
    nodes.add_node(2, 3e-2)
    print('----')
    nodes.print()

    tree.reserve_nodes(2 * n_nodes_reserved)

    nodes.add_node(3, 3e-2)
    nodes.add_node(5, 1.23e-2)
    nodes.add_node(5, 3.5e-2)
    print('----')
    nodes.print()


def bidouille2():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons

    n_samples = 1000
    n_features = 2
    random_state = 123

    X, y = make_moons(n_samples=n_samples, noise=0.25,
                      random_state=random_state)
    X = X.astype('float32')
    y = y.astype('float32')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=42)

    samples = SamplesCollection()
    samples.append(X[:100], y[:100])
    print(samples.features.shape)
    print(samples.labels.shape)
    samples.append(X[100:200], y[100:200])
    print(samples.features.shape)
    print(samples.labels.shape)


def bidouille3():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons

    n_samples = 1000
    n_features = 2
    n_classes = 2
    random_state = 123

    X, y = make_moons(n_samples=n_samples, noise=0.25,
                      random_state=random_state)
    X = X.astype('float32')
    y = y.astype('float32')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=42)

    forest = OnlineForestClassifier(n_classes, n_estimators=1)

    forest.partial_fit(X_train[0:100], y_train[0:100])
    tree = forest.no_python.trees[0]

    for t in range(6):
        # print('-' * 16)
        # print('t:', t)
        # print('x_t:', X_train[t])
        # print('y_t:', y_train[t])
        tree_partial_fit(tree, t)

    # tree_partial_fit(tree, 1)
    # tree_partial_fit(tree, 2)
    # tree_partial_fit(tree, 3)
    # tree_partial_fit(tree, 4)
    # tree_partial_fit(tree, 5)
    #

    print("=" * 32)
    for i in range(tree.nodes.n_nodes):
        node_print(tree, i)

    # forest.partial_fit(X_train[3:6], y_train[3:6])
    # forest.partial_fit(X_train[6:9], y_train[6:9])


def bidouille4():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons

    n_samples = 1000
    n_features = 2
    n_classes = 2
    random_state = 123
    X, y = make_moons(n_samples=n_samples, noise=0.25,
                      random_state=random_state)
    X = X.astype('float32')
    y = y.astype('float32')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=42)

    forest = OnlineForestClassifier(n_classes, n_estimators=1)
    forest.partial_fit(X_train[0:10], y_train[0:10])

    tree = forest.no_python.trees[0]
    samples = forest.no_python.samples

    nodes = tree.nodes
    nodes.add_node(0, 1e-2)
    nodes.add_node(0, 1e-2)
    nodes.add_node(0, 3e-2)
    nodes.add_node(1, 3e-2)
    nodes.add_node(1, 3e-2)
    nodes.add_node(2, 1.23e-2)
    nodes.add_node(4, 3.5e-2)
    nodes.add_node(4, 5.1e-1)
    # tree.nodes.print()

    score = node_score(tree, 3, 1)
    # print('score:', score)

    loss = node_loss(tree, 4, 5)
    # print('loss:', loss)

    node_update_weight(tree, 4, 3)
    node_update_count(tree, 4, 3)
    node_update_downwards(tree, 4, 5, True)
    node_update_weight_tree(tree, 4)

    nodes.left[4] = 6
    nodes.right[4] = 7

    nodes.is_leaf[6] = True
    nodes.is_leaf[7] = True
    # node_print(tree, 4)
    # node_print(tree, 6)
    # node_print(tree, 7)

    node_update_depth(tree, 4, 2)
    # print("node_update_depth(tree, 4, 2)")
    # node_print(tree, 4)
    # node_print(tree, 6)
    # node_print(tree, 7)

    # print("node_is_dirac(tree, 4, 0.):", node_is_dirac(tree, 4, 0.))
    # print("node_is_dirac(tree, 4, 0.):", node_is_dirac(tree, 4, 1.))

    x_t = np.array([-0.1, 0.7], dtype='float32')
    # print("node_get_child(tree, 4, x_t):", node_get_child(tree, 4, x_t))
    # print("node_range(tree, idx_node, j):", node_range(tree, 4, 1))

    extensions = np.empty(n_features, dtype='float32')
    node_compute_range_extension(tree, 4, x_t, extensions)
    # print('extensions:', extensions)

    scores = np.empty(n_classes, dtype='float32')
    node_predict(tree, 4, scores)
    # print('scores:', scores)

    idx_node = 4
    idx_sample = 3

    x_t = tree.samples.features[idx_sample]
    # print('x_t:', x_t)

    tree.nodes.n_samples[idx_node] = 4
    tree.nodes.counts[idx_node, 0] = 2
    tree.nodes.counts[idx_node, 1] = 2

    tree.nodes.time[6] = 3.
    tree.nodes.time[7] = 3.

    split_time = node_compute_split_time(tree, idx_node, idx_sample)
    # print('split_time:', split_time)

    # node_print(tree, idx_node)
    node_split(tree, idx_node, 1.12, 3.14, 0, True)
    # node_print(tree, idx_node)


def bidouille5():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_moons

    n_samples = 1000
    n_features = 2
    n_classes = 2
    random_state = 123
    X, y = make_moons(n_samples=n_samples, noise=0.25,
                      random_state=random_state)
    X = X.astype('float32')
    y = y.astype('float32')
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=42)

    forest = OnlineForestClassifier(n_classes, n_estimators=1)

    n_batches = 10
    batch_size = 32
    for batch in range(0, 3):
        start = batch * batch_size
        end = (batch + 1) * batch_size
        forest.partial_fit(X_train[start:end], y_train[start:end])
        # print('-' * 16)
        # print(X_train[start:end])
        n_samples = forest.no_python.samples.n_samples
        # print(forest.no_python.samples.features[:n_samples])

    df = forest.get_nodes_df(0)

    print(df.head(10))


def bidouille6():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.datasets import make_moons

    n_samples = 1000
    n_features = 2
    max_iter = 40

    X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=0)

    X = MinMaxScaler().fit_transform(X)

    n_classes = int(y.max() + 1)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, stratify=y, test_size=.3, random_state=42)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    of = OnlineForestClassifier(n_classes=n_classes, seed=1234,
                                use_aggregation=True,
                                n_estimators=50, split_pure=True,
                                dirichlet=0.5, step=1.)

    of.partial_fit(X_train, y_train)

    scores = of.predict_proba(X_test)

    # for t in range(0, max_iter + 1):
    #     x_t = X_train[t].reshape(1, n_features).astype('float32')
    #     y_t = np.array([y_train[t]]).astype('float32')

    # X_test = X_test[:10].astype('float32')
    # n_samples_predict = X_test.shape[0]

    # scores = np.zeros((n_samples_predict, n_classes), dtype='float32')
    # of.no_python.predict(X_test, scores)


    # scores = np.empty((n_classes,), dtype='float32')
    # x_t = X_train[max_iter + 1].astype('float32')
    # tree_predict(of.no_python.trees[0], x_t, scores, True)


def bidouille7():
    from sklearn.model_selection import train_test_split

    n_samples = 100000
    n_features = 200
    n_classes = 2
    # max_iter = 40

    X = np.random.randn(n_samples, n_features)

    y = np.random.multinomial(1, [0.5, 0.5], n_samples).argmax(axis=1)

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


    of = OnlineForestClassifier(n_classes=n_classes, seed=1234,
                                use_aggregation=True,
                                n_estimators=20, split_pure=True,
                                dirichlet=0.5, step=1.)

    t1 = time()
    of.partial_fit(X_train, y_train)
    t2 = time()
    print('time fit:', t2 - t1)

    t1 = time()
    scores = of.predict_proba(X_test)
    t2 = time()
    print('time predict:', t2 - t1)


@njit
def bidouille8():
    n_samples = 10000000
    probabilities = np.array([0.1, 0.1, 0.1, 0., 0.2, 0.5], dtype=float32)

    count = np.zeros(probabilities.size)
    for i in range(n_samples):
        j = sample_discrete(probabilities)
        count[j] += 1

    print(count / n_samples)


if __name__ == '__main__':

    bidouille7()

    exit(0)


    # from sklearn.metrics import roc_auc_score
    # import matplotlib.pyplot as plt


    # h = .1
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    #                      np.arange(y_min, y_max, h))
    # Z = np.zeros(xx.shape)
    #
    # cm = plt.cm.RdBu
    # cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    n_estimators = 10

    amf = OnlineForestClassifier(n_classes=2, n_estimators=n_estimators,
                                 seed=123, step=1., use_aggregation=True)

    amf.partial_fit(X_train, y_train)

    print(amf)
