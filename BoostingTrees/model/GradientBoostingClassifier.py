import numpy as np
from typing import Optional, Tuple


#Helper Classes
class TreeNode:
    """
    Single node in a regression tree.

    For leaves we store only `value`; for internal nodes we keep
    `feature_index`, `threshold`, and references to `left` / `right`.
    """
    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
        value: Optional[float] = None,
        is_leaf: bool = False,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

    def predict(self, x: np.ndarray) -> float:
        """Walk the tree until we hit a leaf and return its value."""
        if self.is_leaf:
            return self.value
        if x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)


class DecisionTree:
    """
    Tiny CART-style regression tree — gets fitted to the pseudo-residuals
    produced during boosting.  No pruning, just a depth / sample guard.
    """
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self._rng = np.random.RandomState(random_state)
        self.verbose = verbose
        self.root = None  # will be filled by fit()

    # training helpers
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        """Grow the tree on (X, y)."""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, sample_weight, depth=0)

    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        depth: int,
    ) -> TreeNode:
        """Recursive splitter."""
        n_samples, _ = X.shape
        weighted_mean = np.sum(y * sample_weight) / np.sum(sample_weight)

        # stopping rules
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return TreeNode(value=weighted_mean, is_leaf=True)

        # find best split
        feat, thr, gain = self._find_best_split(X, y, sample_weight)
        if gain < self.min_impurity_decrease:
            return TreeNode(value=weighted_mean, is_leaf=True)

        left_idx = X[:, feat] <= thr
        right_idx = ~left_idx
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return TreeNode(value=weighted_mean, is_leaf=True)

        left = self._grow_tree(X[left_idx], y[left_idx], sample_weight[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], sample_weight[right_idx], depth + 1)
        return TreeNode(feature_index=feat, threshold=thr, left=left, right=right, value=weighted_mean)

    def _find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Brute-force search of (feature, threshold) that maximises MSE reduction."""
        n_samples, n_features = X.shape
        parent_mse = np.average((y - np.average(y, weights=sample_weight)) ** 2, weights=sample_weight)

        best_feat, best_thr, best_gain = 0, 0.0, -np.inf
        feature_subset = (
            self._rng.choice(n_features, self.max_features, replace=False)
            if self.max_features
            else range(n_features)
        )
        if self.verbose:
            print(f"Evaluating {len(feature_subset)} feature(s) at depth search")

        for f in feature_subset:
            vals = np.unique(X[:, f])
            if len(vals) <= 1:
                continue  # nothing to split on
            # continuous feature → mid-points; binary → first value is enough
            thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals) > 2 else [vals[0]]

            for thr in thresholds:
                left = X[:, f] <= thr
                right = ~left
                if not left.any() or not right.any():
                    continue

                l_w, r_w = sample_weight[left], sample_weight[right]
                l_y, r_y = y[left], y[right]

                l_mean = np.average(l_y, weights=l_w)
                r_mean = np.average(r_y, weights=r_w)

                l_mse = np.average((l_y - l_mean) ** 2, weights=l_w)
                r_mse = np.average((r_y - r_mean) ** 2, weights=r_w)

                split_mse = (l_mse * l_w.sum() + r_mse * r_w.sum()) / sample_weight.sum()
                gain = parent_mse - split_mse

                if gain > best_gain:
                    best_feat, best_thr, best_gain = f, thr, gain

        return best_feat, best_thr, best_gain

    # prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorised wrapper around node.predict()."""
        if self.root is None:
            raise ValueError("Tree not fitted yet")
        return np.array([self.root.predict(x) for x in X])



#Gradient Boosting                   
class GradientBoostingClassifier:
    """
    Simple gradient-boosted binary classifier (log-loss with decision-tree base
    learners).  Only {0,1} / {-1,1} targets supported.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample = subsample
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.verbose = verbose

        self._rng = np.random.RandomState(random_state)
        self.trees, self.gammas, self.errors_ = [], [], []
        self.classes_, self.initial_prediction = None, None

    # internal helpers
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Clipped sigmoid to dodge overflow."""
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _to_binary(self, y: np.ndarray) -> np.ndarray:
        """Convert arbitrary 2-class labels to {0,1}."""
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification supported.")
        if not np.all(np.isin(self.classes_, [0, 1, -1])):
            raise ValueError("Labels must be 0/1 or -1/1.")
        return (y == self.classes_[1]).astype(int)

    def _neg_grad(self, y_true: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        """−∂Loss/∂f = y − P(y=1)."""
        return y_true - self._sigmoid(raw_pred)

    def _optimal_gamma(self, y_true, raw_pred, tree_pred):
        """Line-search step for log-loss (closed form)."""
        p = self._sigmoid(raw_pred)
        num = np.sum((y_true - p) * tree_pred)
        den = np.sum(p * (1 - p) * tree_pred**2) + 1e-10
        return num / den

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """Log-odds after all fitted trees."""
        raw = np.full(X.shape[0], self.initial_prediction)
        for g, t in zip(self.gammas, self.trees):
            raw += self.learning_rate * g * t.predict(X)
        return raw

    # API methods
    def fit(self, X: np.ndarray, y: np.ndarray):
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaNs detected – clean your data first.")
        if self.n_estimators < 1:
            raise ValueError("Need at least one tree.")

        y_bin = self._to_binary(y)
        pos, neg = np.sum(y_bin), len(y_bin) - np.sum(y_bin)
        self.initial_prediction = np.log((pos + 1e-15) / (neg + 1e-15))

        raw_pred = np.full(len(y_bin), self.initial_prediction)

        for m in range(self.n_estimators):
            residuals = self._neg_grad(y_bin, raw_pred)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                verbose=self.verbose,
            )

            # row-subsampling if asked
            if self.subsample < 1.0:
                idx = self._rng.choice(len(X), int(len(X) * self.subsample), replace=False)
                tree.fit(X[idx], residuals[idx])
            else:
                tree.fit(X, residuals)

            gamma = self._optimal_gamma(y_bin, raw_pred, tree.predict(X))
            raw_pred += self.learning_rate * gamma * tree.predict(X)

            # bookkeeping
            self.trees.append(tree)
            self.gammas.append(gamma)
            p = self._sigmoid(raw_pred)
            loss = -np.mean(y_bin * np.log(p + 1e-15) + (1 - y_bin) * np.log(1 - p + 1e-15))
            self.errors_.append(loss)

            if self.verbose:
                print(f"Iter {m+1:03d}: loss={loss:.4f}, gamma={gamma:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_predict(X)
        p1 = self._sigmoid(raw)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

    def feature_importances(self) -> np.ndarray:
        """Counts how often each feature is used across all trees."""
        if not self.trees:
            raise ValueError("Model not fitted.")
        counts = np.zeros(self.trees[0].n_features)

        def walk(node):
            if node is None or node.is_leaf:
                return
            counts[node.feature_index] += 1
            walk(node.left)
            walk(node.right)

        for t in self.trees:
            walk(t.root)
        return counts / counts.sum() if counts.sum() else counts

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after each boosting round."""
        raw = np.full(X.shape[0], self.initial_prediction)
        for g, t in zip(self.gammas, self.trees):
            raw += self.learning_rate * g * t.predict(X)
            yield np.where(self._sigmoid(raw) >= 0.5, self.classes_[1], self.classes_[0])
