
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

spec_samples_collection = [
    ('features', float32[:, ::1]),
    ('labels', float32[::1]),
    ('n_samples', uint32)
]



# Sadly there is no function to sample for a discrete distribution in numba
@njit(uint32(float32[::1]))
def sample_discrete(distribution):
    """

    :param distribution:

    Returns
    -------
    output : `uint32`
        Sampled output according to the given distribution

    Notes
    -----
    It is useless to cumsum and searchsorted here, since we want a single sample
    for this distribution (it changes at each call...) so nothing better than
    O(n) can be done here.

    Warning
    -------
    No test is performed here for efficiency: distribution must contain non-
    negative values that sum to one.
    """
    # Notes
    U = uniform(0., 1.)
    cumsum = 0.
    size = distribution.size
    for j in range(size):
        cumsum += distribution[j]
        if U <= cumsum:
            return j
    return size - 1


# TODO: c'est pourri comme strategie si on est vraiment online... faudrait forcer les batch, attendre d'avoir assez de donnees avec de refaire les tableaux

@jitclass(spec_samples_collection)
class SamplesCollection(object):

    def __init__(self):
        self.n_samples = 0

    def append(self, X, y):
        n_samples_batch, n_features = X.shape
        if self.n_samples == 0:
            self.n_samples += n_samples_batch
            # TODO: use copy instead or something else instead ?
            features = np.empty((n_samples_batch, n_features), dtype=float32)
            features[:] = X
            self.features = features
            labels = np.empty(n_samples_batch, dtype=float32)
            labels[:] = y
            self.labels = labels
        else:
            self.n_samples += n_samples_batch
            self.features = np.concatenate((self.features, X))
            self.labels = np.concatenate((self.labels, y))


spec_node_collection = [
    # The index of the node in the tree
    ('index', uint32[::1]),
    # Is the node a leaf ?
    ('is_leaf', boolean[::1]),
    # Depth of the node in the tree
    ('depth', uint8[::1]),
    # Number of samples in the node
    ('n_samples', uint32[::1]),
    # Index of the parent
    ('parent', uint32[::1]),
    # Index of the left child
    ('left', uint32[::1]),
    # Index of the right child
    ('right', uint32[::1]),
    # Index of the feature used for the split
    ('feature', uint32[::1]),
    # The label of the sample saved in the node
    ('y_t', float32[::1]),
    # Logarithm of the aggregation weight for the node
    ('weight', float32[::1]),
    # Logarithm of the aggregation weight for the sub-tree starting at this node
    ('log_weight_tree', float32[::1]),
    # Threshold used for the split
    ('threshold', float32[::1]),
    # Time of creation of the node
    ('time', float32[::1]),
    # Counts the number of sample seen in each class
    ('counts', uint32[:, ::1]),
    # Is the range of the node is memorized ?
    ('memorized', boolean[::1]),
    # Minimum range of the data points in the node
    ('memory_range_min', float32[:, ::1]),
    # Maximum range of the data points in the node
    ('memory_range_max', float32[:, ::1]),
    # List of the samples contained in the range of the node (this allows to
    # compute the range whenever the range memory isn't used)
    # TODO: this is going to be a problem, we don't know in advance how many points end up in the node
    # ('samples', typed.List(uint32)),
    # Number of features
    ('n_features', uint32),
    # Number of classes
    ('n_classes', uint32),
    # Number of nodes actually used
    ('n_nodes', uint32),
    # Number of nodes allocated in memory
    ('n_nodes_reserved', uint32),
    ('n_nodes_computed', uint32)
]


@njit
def resize_array(arr, keep, size, ones=False):
    if arr.ndim == 1:
        if ones:
            new = np.ones((size,), dtype=arr.dtype)
        else:
            new = np.zeros((size,), dtype=arr.dtype)
        new[:keep] = arr[:keep]
        return new
    elif arr.ndim == 2:
        _, n_cols = arr.shape
        new = np.zeros((size, n_cols), dtype=arr.dtype)
        new[:keep] = arr[:keep]
        return new
    else:
        raise ValueError('resize_array can resize only 1D and 2D arrays')


@jitclass(spec_node_collection)
class NodeCollection(object):

    def __init__(self, n_features, n_classes, n_nodes_reserved):
        # print("def __init__(self, n_features, n_classes, n_nodes_reserved):")
        # print('n_features: ', n_features, 'n_classes: ', n_classes,
        #       'n_nodes_reserved:', n_nodes_reserved)

        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: group together arrays of the same type for faster computations ?
        self.index = np.zeros(n_nodes_reserved, dtype=uint32)
        self.is_leaf = np.ones(n_nodes_reserved, dtype=boolean)
        self.depth = np.zeros(n_nodes_reserved, dtype=uint8)
        self.n_samples = np.zeros(n_nodes_reserved, dtype=uint32)
        self.parent = np.zeros(n_nodes_reserved, dtype=uint32)
        self.left = np.zeros(n_nodes_reserved, dtype=uint32)
        self.right = np.zeros(n_nodes_reserved, dtype=uint32)
        self.feature = np.zeros(n_nodes_reserved, dtype=uint32)
        self.y_t = np.zeros(n_nodes_reserved, dtype=float32)
        self.weight = np.zeros(n_nodes_reserved, dtype=float32)
        self.log_weight_tree = np.zeros(n_nodes_reserved, dtype=float32)
        self.threshold = np.zeros(n_nodes_reserved, dtype=float32)
        self.time = np.zeros(n_nodes_reserved, dtype=float32)
        self.counts = np.zeros((n_nodes_reserved, n_classes), dtype=uint32)
        self.memorized = np.zeros(n_nodes_reserved, dtype=boolean)
        self.memory_range_min = np.zeros((n_nodes_reserved, n_features),
                                         dtype=float32)
        self.memory_range_max = np.zeros((n_nodes_reserved, n_features),
                                         dtype=float32)
        # self.samples = typed.List()
        self.n_nodes_reserved = n_nodes_reserved
        self.n_nodes = 0
        self.n_nodes_computed = 0

    def reserve_nodes(self, n_nodes_reserved):
        # print("def reserve_nodes(self, n_nodes_reserved):")
        n_nodes = self.n_nodes
        n_classes = self.n_classes
        n_features = self.n_features

        # print("n_nodes_reserved:", n_nodes_reserved,
        #       "self.n_nodes_reserved:", self.n_nodes_reserved)

        if n_nodes_reserved > self.n_nodes_reserved:
            self.index = resize_array(self.index, n_nodes, n_nodes_reserved)
            # By default, a node is a leaf when newly created
            self.is_leaf = resize_array(self.is_leaf, n_nodes, n_nodes_reserved,
                                        True)
            self.depth = resize_array(self.depth, n_nodes, n_nodes_reserved)
            self.n_samples = resize_array(self.n_samples, n_nodes,
                                          n_nodes_reserved)
            self.parent = resize_array(self.parent, n_nodes, n_nodes_reserved)
            self.left = resize_array(self.left, n_nodes, n_nodes_reserved)
            self.right = resize_array(self.right, n_nodes, n_nodes_reserved)
            self.feature = resize_array(self.feature, n_nodes, n_nodes_reserved)
            self.y_t = resize_array(self.y_t, n_nodes, n_nodes_reserved)
            self.weight = resize_array(self.weight, n_nodes, n_nodes_reserved)
            self.log_weight_tree = resize_array(self.log_weight_tree, n_nodes,
                                                n_nodes_reserved)
            self.threshold = resize_array(self.threshold, n_nodes,
                                          n_nodes_reserved)
            self.time = resize_array(self.time, n_nodes, n_nodes_reserved)
            self.counts = resize_array(self.counts, n_nodes, n_nodes_reserved)
            self.memorized = resize_array(self.memorized, n_nodes,
                                          n_nodes_reserved)
            self.memory_range_min = resize_array(self.memory_range_min, n_nodes,
                                                 n_nodes_reserved)
            self.memory_range_max = resize_array(self.memory_range_max, n_nodes,
                                                 n_nodes_reserved)

        self.n_nodes_reserved = n_nodes_reserved

    def add_node(self, parent, time):
        """
        Add a node with specified parent and creation time. The other fields
        will be provided later on

        Returns
        -------
        output : `int` index of the added node
        """
        if self.n_nodes >= self.n_nodes_reserved:
            raise RuntimeError('self.n_nodes >= self.n_nodes_reserved')
        if self.n_nodes >= self.index.shape[0]:
            raise RuntimeError('self.n_nodes >= self.index.shape[0]')

        node_index = self.n_nodes
        self.index[node_index] = node_index
        self.parent[node_index] = parent
        self.time[node_index] = time
        self.n_nodes += 1
        # This is a new node, so it's necessary computed for now
        self.n_nodes_computed += 1
        return self.n_nodes - 1

    # uint32_t TreeClassifier::add_node(uint32_t parent, float time) {
    #   if (_n_nodes < nodes.size()) {
    #     // We have enough nodes already, so let's use the last free one, and
    #     // just update its time and parent
    #     node(_n_nodes).index(_n_nodes).parent(parent).time(time);
    #     _n_nodes++;
    #     // This node is brand new, so it's necessary computed
    #     _n_nodes_computed++;
    #     return _n_nodes - 1;
    #   } else {
    #     TICK_ERROR("OnlineForest: Something went wrong with nodes allocation !!!")
    #   }
    # }

    def copy_node(self, first, second):
        # We must NOT copy the index
        # self.index[second] = self.index[first]
        self.is_leaf[second] = self.is_leaf[first]
        self.depth[second] = self.depth[first]
        self.n_samples[second] = self.n_samples[first]
        self.parent[second] = self.parent[first]
        self.left[second] = self.left[first]
        self.right[second] = self.right[first]
        self.feature[second] = self.feature[first]
        self.y_t[second] = self.y_t[first]
        self.weight[second] = self.weight[first]
        self.log_weight_tree[second] = self.log_weight_tree[first]
        self.threshold[second] = self.threshold[first]
        self.time[second] = self.time[first]
        self.counts[second, :] = self.counts[first, :]
        self.memorized[second] = self.memorized[first]
        self.memory_range_min[second, :] = self.memory_range_min[first, :]
        self.memory_range_max[second, :] = self.memory_range_max[first, :]

    #   // If the tree isn't full, this is not memorized and node is, then we this to memorize this
    #   if (!_memorized && node._memorized && !_tree.is_full()) {
    #     memorize_range();
    #   } else {
    #     // Warning: range has not been memorized !
    #     _memorized = false;
    #   }
    #   return *this;
    # }

    def print(self):
        n_nodes = self.n_nodes
        n_nodes_reserved = self.n_nodes_reserved
        print('n_nodes_reserved:', n_nodes_reserved, ' n_nodes: ', n_nodes)
        print('index: ', self.index[:n_nodes])
        print('parent: ', self.parent[:n_nodes])
        print('time: ', self.time[:n_nodes])


    # TODO: method qui copie un noeud dans un autre

# NodeClassifier &NodeClassifier::operator=(const NodeClassifier &node) {
#   // We absolutely shall not copy _index !
#   _parent = node._parent;
#   _left = node._left;
#   _right = node._right;
#   _feature = node._feature;
#   _threshold = node._threshold;
#   _time = node._time;
#   _depth = node._depth;
#   _memory_range_min = node._memory_range_min;
#   _memory_range_max = node._memory_range_max;
#   _samples = node._samples;
#   _n_samples = node._n_samples;
#   _y_t = node._y_t;
#   _weight = node._weight;
#   _weight_tree = node._weight_tree;
#   _is_leaf = node._is_leaf;
#   _counts = node._counts;
#


@njit(float32(float32, float32))
def log_sum_2_exp(a, b):
    # Computation of log( (e^a + e^b) / 2) in an overproof way
    # TODO: if |a - b| > 50 skip
    if a > b:
        return a + log((1 + exp(b - a)) / 2)
    else:
        return b + log((1 + exp(a - b)) / 2)

#   // Computation of log( (e^a + e^b) / 2) in an overproof way
#   inline static float log_sum_2_exp(const float a, const float b) {
#     // TODO: if |a - b| > 50 skip
#     if (a > b) {
#       return a + std::log((1 + std::exp(b - a)) / 2);
#     }
#     return b + std::log((1 + std::exp(a - b)) / 2);
#   }


spec_tree_classifier = [
    ('samples', SamplesCollection.class_type.instance_type),
    ('nodes', NodeCollection.class_type.instance_type),
    ('n_features', uint32),
    ('n_classes', uint32),
    ('iteration', uint32),
    ('use_aggregation', boolean),
    ('step', float32),
    ('split_pure', boolean),
    ('intensities', float32[::1]),
    ('dirichlet', float32)
]


@jitclass(spec_tree_classifier)
class TreeClassifier(object):

    def __init__(self, n_features, n_classes, samples):
        self.samples = samples

        default_reserve = 512

        self.nodes = NodeCollection(n_features, n_classes, default_reserve)

        self.n_features = n_features
        self.n_classes = n_classes
        self.iteration = 0

        self.use_aggregation = True
        self.step = 1.
        self.split_pure = False
        self.dirichlet = 0.5
        self.intensities = np.empty(n_features, dtype=float32)

        # We add the root, which is a leaf for now
        self.add_root()

    def add_root(self):
        self.add_node(0, 0.)

    def add_node(self, parent, time):
        return self.nodes.add_node(parent, time)

    def print(self):
        print('Hello from jitclass')
        print(self.nodes.n_nodes_reserved)
        print(self.nodes.n_nodes)


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_score(tree, idx_node, c):
    count = tree.nodes.counts[idx_node, c]
    n_samples = tree.nodes.n_samples[idx_node]
    n_classes = tree.n_classes
    dirichlet = tree.dirichlet
    return (count + dirichlet) / (n_samples + dirichlet * n_classes)

    # float NodeClassifier::score(uint8_t c) const {
    #   // Using the Dirichet prior
    #   return (_counts[c] + dirichlet()) / (_n_samples + dirichlet() * n_classes());
    # }


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_loss(tree, idx_node, idx_sample):
    c = uint8(tree.samples.labels[idx_sample])
    sc = node_score(tree, idx_node, c)
    # print('sc:', sc)
    # TODO: benchmark different logarithms
    return -log(sc)

#     # float NodeClassifier::loss(const float y_t) {
#     #   // Log-loss
#     #   uint8_t c = static_cast<uint8_t>(y_t);
#     #   return -std::log(score(c));
#     # }


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_update_weight(tree, idx_node, idx_sample):
    # print("def node_update_weight(tree, idx_node, idx_sample):")
    loss_t = node_loss(tree, idx_node, idx_sample)
    if tree.use_aggregation:
        tree.nodes.weight[idx_node] -= tree.step * loss_t
    return loss_t


#     # float NodeClassifier::update_weight(const ArrayFloat &x_t, const float y_t) {
#     #   float loss_t = loss(y_t);
#     #   if (use_aggregation()) {
#     #     _weight -= step() * loss_t;
#     #   }
#     #   // We return the loss before updating the predictor of the node in order to
#     #   // update the feature importance in TreeClassifier::go_downwards
#     #   return loss_t;
#     # }


@njit(void(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_update_count(tree, idx_node, idx_sample):
    # TODO: Don't do it twice...
    # print("def node_update_count(tree, idx_node, idx_sample):")
    c = uint32(tree.samples.labels[idx_sample])
    # print('c:', c)
    # print('tree.nodes.counts.shape:', tree.nodes.counts.shape)
    # print('tree.nodes.counts:', tree.nodes.counts)
    tree.nodes.counts[idx_node, c] += 1

#     # void NodeClassifier::update_count(const float y_t) {
#     #   // We update the counts for the class y_t
#     #   _counts[static_cast<uint8_t>(y_t)]++;
#     # }


@njit
def node_print(tree, idx_node):

    print('----------------')
    print(
        'index:', tree.nodes.index[idx_node],
        'depth:', tree.nodes.depth[idx_node],
        'parent:', tree.nodes.parent[idx_node],
        'left:', tree.nodes.left[idx_node],
        'right:', tree.nodes.right[idx_node],
        'is_leaf:', tree.nodes.is_leaf[idx_node],
        'time:', tree.nodes.time[idx_node]
    )

    print(
        'feature:', tree.nodes.feature[idx_node],
        'threshold:', tree.nodes.threshold[idx_node],
        'weight:', tree.nodes.weight[idx_node],
        'log_tree_weight:', tree.nodes.log_weight_tree[idx_node],
    )

    print(
        'n_samples:', tree.nodes.n_samples[idx_node],
        'counts: [', tree.nodes.counts[idx_node, 0], ',',
        tree.nodes.counts[idx_node, 1], ']',
        'memorized:', tree.nodes.memorized[idx_node],
        'memory_range_min: [', tree.nodes.memory_range_min[idx_node, 0], ',',
        tree.nodes.memory_range_min[idx_node, 1], ']',
        'memory_range_max: [', tree.nodes.memory_range_max[idx_node, 0], ',',
        tree.nodes.memory_range_max[idx_node, 1], ']'
    )

    # # self.samples = typed.List()
    # self.n_nodes_reserved = n_nodes_reserved
    # self.n_nodes = 0
    # self.n_nodes_computed = 0


@njit(void(TreeClassifier.class_type.instance_type, uint32, uint32, boolean))
def node_update_downwards(tree, idx_node, idx_sample, do_update_weight):
    # print('node_update_downwards(self, idx_node, idx_sample, do_update_weight)')
    x_t = tree.samples.features[idx_sample]
    y_t = tree.samples.labels[idx_sample]
    memory_range_min = tree.nodes.memory_range_min[idx_node]
    memory_range_max = tree.nodes.memory_range_max[idx_node]

    # If it is the first sample, we copy the features vector into the min and
    # max range
    if tree.nodes.n_samples[idx_node] == 0:
        for j in range(tree.n_features):
            x_tj = x_t[j]
            memory_range_min[j] = x_tj
            memory_range_max[j] = x_tj
    # Otherwise, we update the range
    else:
        for j in range(tree.n_features):
            x_tj = x_t[j]
            if x_tj < memory_range_min[j]:
                memory_range_min[j] = x_tj
            if x_tj > memory_range_max[j]:
                memory_range_max[j] = x_tj

    # TODO: we should save the sample here and do a bunch of stuff about memorization
    #   // Now, we actually add the sample
    #   _samples.emplace_back(sample);
    #   // We node range is memorized, then we need to inform as well the tree
    #   if (_memorized) {
    #     _tree.incr_n_samples(_n_samples, _index);
    #   }

    # One more sample in the node
    tree.nodes.n_samples[idx_node] += 1

    if do_update_weight:
        # TODO: Using x_t and y_t should be better...
        node_update_weight(tree, idx_node, idx_sample)

    node_update_count(tree, idx_node, idx_sample)
    #   // Now that the node is updated, we update also its range type
    #   _tree.update_range_type(_n_samples, _index);

    # TODO: actually do_update_weight is always false if the node is a leaf, otherwise it's true
    # void NodeClassifier::update_downwards(uint32_t sample, bool do_update_weight) {
    #   // Get the sample features and label
    #   const ArrayFloat& x_t = sample_features(sample);
    #   float y_t = sample_label(sample);
    #   if (_memorized) {
    #     // If range is memorized, we update it
    #     for (ulong j = 0; j < n_features(); ++j) {
    #       float x_tj = x_t[j];
    #       if (x_tj < _memory_range_min[j]) {
    #         _memory_range_min[j] = x_tj;
    #       }
    #       if (x_tj > _memory_range_max[j]) {
    #         _memory_range_max[j] = x_tj;
    #       }
    #     }
    #   }
    #   // Now, we actually add the sample
    #   _samples.emplace_back(sample);
    #   // We node range is memorized, then we need to inform as well the tree
    #   if (_memorized) {
    #     _tree.incr_n_samples(_n_samples, _index);
    #   }
    #   _n_samples++;
    #   if (do_update_weight) {
    #     update_weight(x_t, y_t);
    #   }
    #   update_count(y_t);
    #   // Now that the node is updated, we update also its range type
    #   _tree.update_range_type(_n_samples, _index);
    # }


@njit(void(TreeClassifier.class_type.instance_type, uint32))
def node_update_weight_tree(tree, idx_node):
    if tree.nodes.is_leaf[idx_node]:
        tree.nodes.log_weight_tree[idx_node] = tree.nodes.weight[idx_node]
    else:
        left = tree.nodes.left[idx_node]
        right = tree.nodes.right[idx_node]
        weight = tree.nodes.weight[idx_node]
        log_weight_tree = tree.nodes.log_weight_tree
        tree.nodes.weight[idx_node] = log_sum_2_exp(
            weight, log_weight_tree[left] + log_weight_tree[right]
        )

# void NodeClassifier::update_weight_tree() {
#   if (_is_leaf) {
#     _weight_tree = _weight;
#   } else {
#     _weight_tree = log_sum_2_exp(_weight, node(_left).weight_tree() + node(_right).weight_tree());
#   }
# }
#


@njit(void(TreeClassifier.class_type.instance_type, uint32, uint8))
def node_update_depth(tree, idx_node, depth):
    # print("def node_update_depth(tree, idx_node, depth):")
    depth += 1
    tree.nodes.depth[idx_node] = depth
    if tree.nodes.is_leaf[idx_node]:
        return
    else:
        left = tree.nodes.left[idx_node]
        right = tree.nodes.right[idx_node]
        node_update_depth(tree, left, depth)
        node_update_depth(tree, right, depth)

# # void TreeClassifier::update_depth(uint32_t node_index, uint8_t depth) {
#     #   NodeClassifier &current_node = node(node_index);
#     #   depth++;
#     #   current_node.depth(depth);
#     #   if (current_node.is_leaf()) {
#     #     return;
#     #   } else {
#     #     update_depth(current_node.left(), depth);
#     #     update_depth(current_node.right(), depth);
#     #   }
#     # }


@njit(boolean(TreeClassifier.class_type.instance_type, uint32, float32))
def node_is_dirac(tree, idx_node, y_t):
    c = uint8(y_t)
    n_samples = tree.nodes.n_samples[idx_node]
    count = tree.nodes.counts[idx_node, c]
    return n_samples == count

# bool NodeClassifier::is_dirac(const float y_t) {
#   // Returns true if the node only has the same label as y_t
#   return (_n_samples == _counts[static_cast<uint8_t>(y_t)]);
# }

@njit(uint32(TreeClassifier.class_type.instance_type, uint32, float32[::1]))
def node_get_child(tree, idx_node, x_t):
    feature = tree.nodes.feature[idx_node]
    threshold = tree.nodes.threshold[idx_node]
    if x_t[feature] <= threshold:
        return tree.nodes.left[idx_node]
    else:
        return tree.nodes.right[idx_node]

# uint32_t NodeClassifier::get_child(const ArrayFloat &x_t) {
#   if (x_t[_feature] <= _threshold) {
#     return _left;
#   } else {
#     return _right;
#   }
# }


@njit(Tuple((float32, float32))(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_range(tree, idx_node, j):
    if tree.nodes.n_samples[idx_node] == 0:
        raise RuntimeError("Node has no range since it has no samples")
    else:
        return tree.nodes.memory_range_min[idx_node, j], \
               tree.nodes.memory_range_max[idx_node, j]

# TODO: do the version without memory...
# std::pair<float, float> NodeClassifier::range(uint32_t j) const {
#   if (_n_samples == 0) {
#     // TODO: weird to have to do this
#     TICK_ERROR("Node has no range since it has no samples")
#   } else {
#     if (_memorized) {
#       // If range is memorized, then we return it directly
#       return std::pair<float, float>(_memory_range_min[j], _memory_range_max[j]);
#     } else {
#       // Otherwise, we need to compute it from the samples features
#       if (_n_samples == 1) {
#         uint32_t sample = _samples.front();
#         float feature_j = sample_features(sample)[j];
#         return std::pair<float, float>(feature_j, feature_j);
#       } else {
#         float x_0j = sample_features(_samples.front())[j];
#         float range_min = x_0j;
#         float range_max = x_0j;
#         for(auto sample_iter = std::next(_samples.begin()); sample_iter != _samples.end(); ++sample_iter) {
#           float x_tj = sample_features(*sample_iter)[j];
#           if (x_tj < range_min) {
#             range_min = x_tj;
#           }
#           if (x_tj > range_max) {
#             range_max = x_tj;
#           }
#         }
#         return std::pair<float, float>(range_min, range_max);
#       }
#     }
#   }
# }


@njit(float32(TreeClassifier.class_type.instance_type, uint32, float32[::1],
              float32[::1]))
def node_compute_range_extension(tree, idx_node, x_t, extensions):
    extensions_sum = 0
    for j in range(tree.n_features):
        x_tj = x_t[j]
        feature_min_j, feature_max_j = node_range(tree, idx_node, j)
        if x_tj < feature_min_j:
            diff = feature_min_j - x_tj
        elif x_tj > feature_max_j:
            diff = x_tj - feature_max_j
        else:
            diff = 0
        extensions[j] = diff
        extensions_sum += diff
    return extensions_sum

# void NodeClassifier::compute_range_extension(const ArrayFloat &x_t, ArrayFloat &extensions,
#                                              float &extensions_sum, float &extensions_max) {
#   extensions_sum = 0;
#   extensions_max = std::numeric_limits<float>::lowest();
#   for (uint32_t j = 0; j < n_features(); ++j) {
#     float x_tj = x_t[j];
#     float feature_min_j, feature_max_j;
#     std::tie(feature_min_j, feature_max_j) = range(j);
#     float diff;
#     if (x_tj < feature_min_j) {
#       diff = feature_min_j - x_tj;
#     } else {
#       if (x_tj > feature_max_j) {
#         diff = x_tj - feature_max_j;
#       } else {
#         diff = 0;
#       }
#     }
#     extensions[j] = diff;
#     extensions_sum += diff;
#     if (diff > extensions_max) {
#       extensions_max = diff;
#     }
#   }
# }


@njit(void(TreeClassifier.class_type.instance_type, uint32, float32[::1]))
def node_predict(tree, idx_node, scores):
    # TODO: this is a bit silly ?... do everything at once
    for c in range(tree.n_classes):
        scores[c] = node_score(tree, idx_node, c)


# void NodeClassifier::predict(ArrayFloat &scores) const {
#   for (uint8_t c = 0; c < n_classes(); ++c) {
#     scores[c] = score(c);
#   }
# }
#

#     def node_forget_range(self):
#         pass
#
# void NodeClassifier::forget_range() {
#   if (_memorized) {
#     _memory_range_min.clear();
#     _memory_range_max.clear();
#     // Don't forget to inform the tree about this node
#     _tree.remove_from_disposables(_n_samples, _index);
#     _memorized = false;
#   }
# }
#
#     def node_memorize_range(self):
#         pass
#
# void NodeClassifier::memorize_range() {
#   if(!_memorized && _n_samples >= 2 && !_tree.is_full()) {
#     // We memorize range if it is not memorized yet, if it has more than one sample and if the
#     // tree is not full
#     if (_memory_range_min.empty()) {
#       // TODO: new one or reserve ?
#       // _memory_range_min.reserve(n_features());
#       _memory_range_min = std::vector<float>(n_features());
#       _memory_range_max = std::vector<float>(n_features());
#     }
#     if(!_samples.empty()) {
#       // First, copy the first sample into the range
#       float *begin = sample_features(_samples.front()).data();
#       float *end = begin + n_features();
#       std::copy(begin, end, _memory_range_min.begin());
#       std::copy(begin, end, _memory_range_max.begin());
#       // Then, update the range using the rest of the points
#       for(auto sample_iter= std::next(_samples.begin()); sample_iter != _samples.end(); ++sample_iter) {
#         const ArrayFloat & x_t = sample_features(*sample_iter);
#         for (uint32_t j=0; j < n_features(); ++j) {
#           float x_tj = x_t[j];
#           if (x_tj < _memory_range_min[j]) {
#             _memory_range_min[j] = x_tj;
#           }
#           if (x_tj > _memory_range_max[j]) {
#             _memory_range_max[j] = x_tj;
#           }
#         }
#       }
#     }
#     _memorized = true;
#     // Don't forget to inform the tree about this node
#     _tree.add_to_disposables(_n_samples, _index);
#   }
# }


@njit(float32(TreeClassifier.class_type.instance_type, uint32, uint32))
def node_compute_split_time(tree, idx_node, idx_sample):
    # print("node_compute_split_time(self, idx_node, idx_sample)")
    y_t = tree.samples.labels[idx_sample]

    #  Don't split if the node is pure: all labels are equal to the one of y_t
    # TODO: mais si idx_node est root on renvoie 0 forcement
    if tree.split_pure and node_is_dirac(tree, idx_node, y_t):
        # print("tree.split_pure and node_is_dirac(tree, idx_node, y_t)")
        return 0

    x_t = tree.samples.features[idx_sample]

    extensions_sum = node_compute_range_extension(tree, idx_node, x_t,
                                                  tree.intensities)

    # print('extensions_sum:', extensions_sum)

    # If x_t extends the current range of the node
    if extensions_sum > 0:
        # Sample an exponential with intensity = extensions_sum
        T = expovariate(extensions_sum)
        # print("T:", T)
        time = tree.nodes.time[idx_node]
        # Splitting time of the node (if splitting occurs)
        split_time = time + T

        # print('split_time:', split_time)

        # If the node is a leaf we must split it
        if tree.nodes.is_leaf[idx_node]:
            # print("if tree.nodes.is_leaf[idx_node]:")
            return split_time
        # Otherwise we apply Mondrian process dark magic :)

        # 1. We get the creation time of the childs (left and right is the
        #    same)
        left = tree.nodes.left[idx_node]
        child_time = tree.nodes.time[left]
        # 2. We check if splitting time occurs before child creation time
        # print("split_time < child_time:", split_time, "<", child_time)
        if split_time < child_time:
            return split_time

    return 0


#         # float TreeClassifier::compute_split_time(uint32_t node_index, uint32_t sample) {
#         #   NodeClassifier &current_node = node(node_index);
#         #   float y_t = sample_label(sample);
#         #   // Don't split if the node is pure: all labels are equal to the one of y_t
#         #   if (!forest.split_pure() && current_node.is_dirac(y_t)) {
#         #     return 0;
#         #   }
#         #
#         #   // Don't split if the number of samples in the node is not large enough
#         #   /*
#         #   uint32_t min_samples_split = forest.min_samples_split();
#         #   if ((min_samples_split > 0) && (current_node.n_samples() < min_samples_split)) {
#         #     return 0;
#         #   }
#         #
#         #
#         #   // Get maximum number of nodes allowed in the tree
#         #   uint32_t max_nodes = forest.max_nodes();
#         #   // Don't split if there is already too many nodes
#         #   if ((max_nodes > 0) && (_n_nodes >= max_nodes)) {
#         #     return 0;
#         #   }
#         #   */
#         #
#         #   // Let's compute the extension of the range of the current node, and its sum
#         #   float extensions_sum, extensions_max;
#         #
#         #   const ArrayFloat& x_t = sample_features(sample);
#         #   current_node.compute_range_extension(x_t, intensities, extensions_sum, extensions_max);
#         #
#         #   // Get the min extension size
#         # /*
#         #   float min_extension_size = forest.min_extension_size();
#         #   // Don't split if the extension is too small
#         #   if ((min_extension_size > 0) && extensions_max < min_extension_size) {
#         #     return 0;
#         #   }
#         # */
#         #
#         #   // If the sample x_t extends the current range of the node
#         #   if (extensions_sum > 0) {
#         #     // TODO: check that intensity is indeed intensity in the rand.h
#         #     float T = forest.sample_exponential(extensions_sum);
#         #     float time = current_node.time();
#         #     // Splitting time of the node (if splitting occurs)
#         #     float split_time = time + T;
#         #     if (current_node.is_leaf()) {
#         #       // If the node is a leaf we must split it
#         #       return split_time;
#         #     }
#         #     // Otherwise we apply Mondrian process dark magic :)
#         #     // 1. We get the creation time of the childs (left and right is the same)
#         #     float child_time = node(current_node.left()).time();
#         #     // 2. We check if splitting time occurs before child creation time
#         #     // Sample a exponential random variable with intensity
#         #     if (split_time < child_time) {
#         #       // std::cout << "Done with TreeClassifier::compute_split_time returning
#         #       // " << split_time << std::endl;
#         #       return split_time;
#         #     }
#         #   }
#         #   return 0;
#         # }


@njit(void(TreeClassifier.class_type.instance_type, uint32, float32, float32,
           uint32, boolean))
def node_split(tree, idx_node, split_time, threshold, feature,
               is_right_extension):
    # print("def node_split(tree, idx_node, split_time, threshold, feature,")
    # Create the two splits
    left_new = tree.nodes.add_node(idx_node, split_time)
    right_new = tree.nodes.add_node(idx_node, split_time)
    # TODO: left_new and right_new nodes don't use memory yet

    if is_right_extension:
        # left_new is the same as idx_node, excepted for the parent, time
        # and the fact that it's a leaf
        tree.nodes.copy_node(idx_node, left_new)

        # so we need to put back the correct parent and time
        tree.nodes.parent[left_new] = idx_node
        tree.nodes.time[left_new] = split_time
        # right_new must have idx_node has parent
        tree.nodes.parent[right_new] = idx_node
        tree.nodes.time[right_new] = split_time

        # We must tell the old childs that they have a new parent, if the
        # current node is not a leaf
        if not tree.nodes.is_leaf[idx_node]:
            left = tree.nodes.left[idx_node]
            right = tree.nodes.right[idx_node]
            tree.nodes.parent[left] = left_new
            tree.nodes.parent[right] = left_new
    else:
        tree.nodes.copy_node(idx_node, right_new)
        tree.nodes.parent[right_new] = idx_node
        tree.nodes.time[right_new] = split_time
        tree.nodes.parent[left_new] = idx_node
        tree.nodes.time[left_new] = split_time
        if not tree.nodes.is_leaf[idx_node]:
            left = tree.nodes.left[idx_node]
            right = tree.nodes.right[idx_node]
            tree.nodes.parent[left] = right_new
            tree.nodes.parent[right] = right_new

    tree.nodes.feature[idx_node] = feature
    tree.nodes.threshold[idx_node] = threshold
    tree.nodes.left[idx_node] = left_new
    tree.nodes.right[idx_node] = right_new
    tree.nodes.is_leaf[idx_node] = False


@njit(uint32(TreeClassifier.class_type.instance_type, uint32))
def tree_go_downwards(tree, idx_sample):
    # print("----------------")
    # print('def go_downwards(self, idx_sample):')
    # We update the nodes along the path which leads to the leaf containing
    # x_t. For each node on the path, we consider the possibility of
    # splitting it, following the Mondrian process definition.

    # Index of the root is 0
    idx_current_node = 0
    x_t = tree.samples.features[idx_sample]

    if tree.iteration == 0:
        # If it's the first iteration, we just put x_t in the range of root
        # print("if tree.iteration == 0")
        node_update_downwards(tree, idx_current_node, idx_sample, False)
        return idx_current_node
    else:
        while True:
            # If it's not the first iteration (otherwise the current node
            # is root with no range), we consider the possibility of a split
            split_time = node_compute_split_time(tree, idx_current_node,
                                                 idx_sample)
            # print('split_time:', split_time)
            if split_time > 0:
                # We split the current node: because the current node is a
                # leaf, or because we add a new node along the path
                # We normalize the range extensions to get probabilities
                tree.intensities[tree.intensities == 0] = 1e-6
                tree.intensities /= (tree.intensities.sum() + 1e-6)
                # Sample the feature at random with with a probability
                # proportional to the range extensions
                # TODO: faster implem
                # print("before")
                # print("tree.intensities:", tree.intensities)
                feature = sample_discrete(tree.intensities)
                # feature = uint32(multinomial(1, tree.intensities).argmax())
                # print("after")
                # print("feature:", feature)
                x_tf = x_t[feature]
                # Is it a right extension of the node ?
                range_min, range_max = node_range(tree, idx_current_node,
                                                  feature)
                is_right_extension = x_tf > range_max
                if is_right_extension:
                    threshold = uniform(range_max, x_tf)
                else:
                    threshold = uniform(x_tf, range_min)

                node_split(tree, idx_current_node, split_time, threshold,
                           feature, is_right_extension)

                # Update the current node
                node_update_downwards(tree, idx_current_node, idx_sample, True)

                left = tree.nodes.left[idx_current_node]
                right = tree.nodes.right[idx_current_node]
                depth = tree.nodes.depth[idx_current_node]

                # Now, get the next node
                if is_right_extension:
                    idx_current_node = right
                else:
                    idx_current_node = left

                node_update_depth(tree, left, depth)
                node_update_depth(tree, right, depth)

                # This is the leaf containing the sample point (we've just
                # splitted the current node with the data point)
                leaf = idx_current_node
                node_update_downwards(tree, leaf, idx_sample, False)
                # print('leaf:', leaf)
                return leaf
            else:
                # There is no split, so we just update the node and go to
                # the next one
                node_update_downwards(tree, idx_current_node, idx_sample, True)
                is_leaf = tree.nodes.is_leaf[idx_current_node]
                if is_leaf:
                    return idx_current_node
                else:
                    idx_current_node = node_get_child(tree, idx_current_node, x_t)


@njit(void(TreeClassifier.class_type.instance_type, uint32))
def tree_go_upwards(tree, leaf):
    idx_current_node = leaf
    if tree.iteration >= 1:
        while True:
            node_update_weight_tree(tree, idx_current_node)
            if idx_current_node == 0:
                # We arrived at the root
                break
            # Note that the root node is updated as well
            # We go up to the root in the tree
            idx_current_node = tree.nodes.parent[idx_current_node]


# void TreeClassifier::go_upwards(uint32_t leaf_index) {
#   uint32_t current = leaf_index;
#   if (iteration >= 1) {
#     while (true) {
#       NodeClassifier &current_node = node(current);
#       current_node.update_weight_tree();
#       if (current == 0) {
#         // We arrive at the root
#         break;
#       }
#       // We must update the root node
#       current = node(current).parent();
#     }
#   }
# }


@njit(void(TreeClassifier.class_type.instance_type, uint32))
def tree_partial_fit(tree, idx_sample):
    leaf = tree_go_downwards(tree, idx_sample)
    if tree.use_aggregation:
        tree_go_upwards(tree, leaf)
    tree.iteration += 1


# # void TreeClassifier::fit(uint32_t sample) {
# #   uint32_t leaf = go_downwards(sample);
# #   if (use_aggregation()) {
# #     go_upwards(leaf);
# #   }
# #   iteration++;
# # }

@njit(uint32(TreeClassifier.class_type.instance_type, float32[::1]))
def tree_get_leaf(tree, x_t):
    # Find the index of the leaf that contains the sample. Start at the root.
    # Index of the root is 0
    node = 0
    is_leaf = False
    while not is_leaf:
        is_leaf = tree.nodes.is_leaf[node]
        if not is_leaf:
            feature = tree.nodes.feature[node]
            threshold = tree.nodes.threshold[node]
            if x_t[feature] <= threshold:
                node = tree.nodes.left[node]
            else:
                node = tree.nodes.right[node]
    return node


# uint32_t TreeClassifier::get_leaf(const ArrayFloat &x_t) {
#   // Find the index of the leaf that contains the sample. Start at the root.
#   // Index of the root is 0
#   uint32_t index_current_node = 0;
#   bool is_leaf = false;
#   while (!is_leaf) {
#     NodeClassifier &current_node = node(index_current_node);
#     is_leaf = current_node.is_leaf();
#     if (!is_leaf) {
#       uint32_t feature = current_node.feature();
#       float threshold = current_node.threshold();
#       if (x_t[feature] <= threshold) {
#         index_current_node = current_node.left();
#       } else {
#         index_current_node = current_node.right();
#       }
#     }
#   }
#   return index_current_node;
# }


@njit(void(TreeClassifier.class_type.instance_type, float32[::1], float32[::1], boolean))
def tree_predict(tree, x_t, scores, use_aggregation):
    leaf = tree_get_leaf(tree, x_t)
    # print('leaf:', leaf)
    if not use_aggregation:
        node_predict(tree, leaf, scores)
        return
    current = leaf
    pred_new = np.empty(tree.n_classes, float32)
    while True:
        if tree.nodes.is_leaf[current]:
            node_predict(tree, current, scores)
        else:
            weight = tree.nodes.weight[current]
            log_weight_tree = tree.nodes.log_weight_tree[current]
            w = exp(weight - log_weight_tree)
            # Get the predictions of the current node
            node_predict(tree, current, pred_new)
            for c in range(tree.n_classes):
                scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c]
        # Root must be update as well
        if current == 0:
            break
        # And now we go up
        current = tree.nodes.parent[current]


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
        trees = [TreeClassifier(n_features, n_classes, samples) for _ in range(n_estimators)]
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
            raise RuntimeError("You must call ``partial_fit`` before ``predict``.")


# void OnlineForestClassifier::predict(const SArrayFloat2dPtr features, SArrayFloat2dPtr scores) {
#   scores->fill(0.);
#   uint32_t n_features = static_cast<uint32_t>(features->n_cols());
#   check_n_features(n_features, true);
#   if (_iteration > 0) {
#     uint32_t n_samples = static_cast<uint32_t>(features->n_rows());
#     ArrayFloat scores_tree(_n_classes);
#     scores_tree.fill(0.);
#     ArrayFloat scores_forest(_n_classes);
#     scores_forest.fill(0.);
#     for (uint32_t i = 0; i < n_samples; ++i) {
#       // The prediction is simply the average of the predictions
#       ArrayFloat scores_i = view_row(*scores, i);
#       for (TreeClassifier &tree : trees) {
#         tree.predict(view_row(*features, i), scores_tree, _use_aggregation);
#         // TODO: use a .incr method instead ??
#         scores_i.mult_incr(scores_tree, 1.);
#       }
#       scores_i /= _n_trees;
#     }
#   } else {
#     TICK_ERROR("You must call ``fit`` before ``predict``.")
#   }
# }


class OnlineForestClassifier(object):

    def __init__(self, n_classes, n_estimators: int = 10, step: float = 1.,
                     criterion: str = 'log', use_aggregation: bool = True,
                     dirichlet: float = None, split_pure: bool = False,
                     min_samples_split: int=-1, max_features: int=-1,
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
