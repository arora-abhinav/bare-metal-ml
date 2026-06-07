import math
import heapq


# ── distance metrics ──────────────────────────────────────────────────────────

def euclidean(x1, x2):
    res = 0
    for i in range(len(x1)):
        res += (x1[i] - x2[i]) ** 2
    return math.sqrt(res)

def manhattan(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i] - x2[i])
    return dist

def cosine(x1, x2):
    dot_prod = 0
    x1_len = 0
    x2_len = 0
    for i in range(len(x1)):
        dot_prod += x1[i] * x2[i]
        x1_len += x1[i] ** 2
        x2_len += x2[i] ** 2
    return 1 - dot_prod / (math.sqrt(x1_len) * math.sqrt(x2_len))


# ── KNN ───────────────────────────────────────────────────────────────────────

class KNN:
    """
    K-Nearest Neighbours classifier.

    Uses a max-heap of size k to track the k nearest neighbours seen so far,
    swapping out the current worst when a closer point is found.

    Parameters
    ----------
    k : int
        Number of neighbours.
    metric : callable
        Distance function. Defaults to Euclidean.

    Example
    -------
    >>> from bare_metal_ml.knn import KNN, euclidean
    >>> model = KNN(k=5, metric=euclidean)
    >>> model.fit(x_train, y_train)
    >>> print(model.accuracy(x_test, y_test))
    """

    def __init__(self, k=5, metric=None):
        self.k = k
        self.metric = metric if metric is not None else euclidean
        self._x = None
        self._y = None

    def fit(self, x_train, y_train):
        self._x = x_train
        self._y = y_train

    def predict_one(self, testing_vector):
        # Max-heap of size k — stores (-dist, index, label) so heapq (min-heap)
        # pops the largest distance when the heap is full
        heap = []
        for i in range(len(self._x)):
            dist = self.metric(self._x[i], testing_vector)
            if len(heap) < self.k:
                heapq.heappush(heap, (-dist, i, self._y[i]))
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, i, self._y[i]))

        votes = {}
        for i in range(len(heap)):
            label = heap[i][2]
            if label not in votes:
                votes[label] = 1
            else:
                votes[label] += 1

        sorted_map = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        return sorted_map[0][0]

    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            predictions.append(self.predict_one(x_test[i]))
        return predictions

    def accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return (correct / len(y_test)) * 100.0


# ── KD-Tree ───────────────────────────────────────────────────────────────────

class Node:
    def __init__(self, median, coordinate, index):
        self.left = None
        self.right = None
        # median stores (dimension_index, median_value) — the splitting decision
        self.median = median
        self.coordinate = coordinate
        self.index = index


class KDTree:
    """
    KD-Tree for accelerated nearest-neighbour search.

    Recursively partitions the feature space by alternating the splitting
    dimension at each level. During search, entire subtrees are pruned when
    the perpendicular distance to the splitting hyperplane exceeds the current
    best, making lookups much faster than brute-force KNN on large datasets.

    Example
    -------
    >>> from bare_metal_ml.knn import KDTree
    >>> tree = KDTree()
    >>> tree.fit(x_train, y_train)
    >>> print(tree.accuracy(x_test, y_test, k=9))
    """

    def __init__(self):
        self._root = None
        self._y_train = None
        self._n_features = None

    def fit(self, x_train, y_train):
        self._y_train = y_train
        self._n_features = len(x_train[0])
        training_examples = [(list(x_train[i]), i) for i in range(len(x_train))]
        self._root = self._construct(training_examples, index=0)

    def _obtain_median(self, coords):
        median_index = int((len(coords) - 1) / 2)
        return coords[median_index], median_index

    def _construct(self, training_examples, index):
        if len(training_examples) == 0:
            return None
        sorted_coords = sorted(training_examples, key=lambda x: x[0][index])
        median, median_index = self._obtain_median(sorted_coords)
        node = Node(
            median=(index, median[0][index]),
            coordinate=median[0],
            index=median[1]
        )
        if len(training_examples) > 1:
            node.left = self._construct(sorted_coords[:median_index], (index + 1) % self._n_features)
            node.right = self._construct(sorted_coords[median_index + 1:], (index + 1) % self._n_features)
        return node

    def _search(self, node, query, ind, best_dist, heap, k):
        if node is None:
            return best_dist, heap

        current_dist = euclidean(node.coordinate, query)

        if len(heap) < k:
            heapq.heappush(heap, (-current_dist, node.index, node))
            best_dist = -heap[0][0]

        if len(heap) == k and best_dist > current_dist:
            heapq.heapreplace(heap, (-current_dist, node.index, node))
            best_dist = -heap[0][0]

        # Descend the side the query falls on first
        if query[ind] > node.median[1]:
            primary = node.right
            other = node.left
        else:
            primary = node.left
            other = node.right

        best_dist, heap = self._search(primary, query, (ind + 1) % self._n_features, best_dist, heap, k)

        # Explore the other side only if the splitting plane is within best_dist
        if abs(query[ind] - node.median[1]) < best_dist:
            best_dist, heap = self._search(other, query, (ind + 1) % self._n_features, best_dist, heap, k)

        return best_dist, heap

    def predict_one(self, query, k=1):
        heap = []
        best_dist = float('inf')
        best_dist, heap = self._search(self._root, query, 0, best_dist, heap, k)

        votes = {}
        for dist, index, node in heap:
            label = self._y_train[node.index]
            if label not in votes:
                votes[label] = 1
            else:
                votes[label] += 1

        return max(votes, key=lambda x: votes[x])

    def predict(self, x_test, k=1):
        predictions = []
        for i in range(len(x_test)):
            predictions.append(self.predict_one(x_test[i], k))
        return predictions

    def accuracy(self, x_test, y_test, k=1):
        predictions = self.predict(x_test, k)
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                correct += 1
        return (correct / len(y_test)) * 100.0
