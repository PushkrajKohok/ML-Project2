import numpy as np
from typing import Optional, List, Tuple, Union

class TreeNode:
    """
    A node in a decision tree for gradient boosting.
    """
    def __init__(self, 
                feature_index: Optional[int] = None,
                threshold: Optional[float] = None,
                left: Optional['TreeNode'] = None,
                right: Optional['TreeNode'] = None,
                value: Optional[float] = None,
                is_leaf: bool = False):
        """
        Initialize a decision tree node.
        
        Args:
            feature_index: Index of the feature to split on
            threshold: Threshold value for the split
            left: Left child node
            right: Right child node
            value: The output value for this node (if leaf)
            is_leaf: Whether this node is a leaf node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf
    
    def predict(self, x: np.ndarray) -> float:
        """
        Predict the value for a single sample.
        
        Args:
            x: Single sample features
            
        Returns:
            Predicted value
        """
        if self.is_leaf:
            return self.value
        
        if x[self.feature_index] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class DecisionTree:
    """
    A regression decision tree for use in gradient boosting.
    """
    def __init__(self, 
                max_depth: int = 3, 
                min_samples_split: int = 2,
                min_impurity_decrease: float = 0.0):
        """
        Initialize a decision tree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum number of samples required to split a node
            min_impurity_decrease: Minimum decrease in impurity required for split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Build a decision tree from the training set (X, y).
        
        Args:
            X: Training features
            y: Target values (residuals)
            sample_weight: Sample weights
        """
        if sample_weight is None:
            sample_weight = np.ones(len(y))
            
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y, sample_weight, depth=0)
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, depth: int) -> TreeNode:
        """
        Recursively build a decision tree by splitting data.
        
        Args:
            X: Training features
            y: Target values (residuals)
            sample_weight: Sample weights
            depth: Current depth of the tree
            
        Returns:
            Root node of the decision tree
        """
        n_samples, n_features = X.shape
        
        # Calculate weighted mean of target values
        weighted_sum = np.sum(y * sample_weight)
        total_weight = np.sum(sample_weight)
        
        if total_weight > 0:
            node_value = weighted_sum / total_weight
        else:
            node_value = 0.0
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.all(np.abs(y - y[0]) < 1e-10)):  # All y values are the same
            return TreeNode(value=node_value, is_leaf=True)
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, sample_weight)
        
        # If the gain is too small, make this a leaf node
        if best_gain < self.min_impurity_decrease:
            return TreeNode(value=node_value, is_leaf=True)
        
        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check if the split is meaningful
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return TreeNode(value=node_value, is_leaf=True)
        
        # Recursively build the left and right subtrees
        left_subtree = self._grow_tree(
            X[left_indices], 
            y[left_indices], 
            sample_weight[left_indices], 
            depth + 1
        )
        
        right_subtree = self._grow_tree(
            X[right_indices], 
            y[right_indices], 
            sample_weight[right_indices], 
            depth + 1
        )
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
            value=node_value
        )
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> Tuple[int, float, float]:
        """
        Find the best feature and threshold for splitting.
        
        Args:
            X: Training features
            y: Target values (residuals)
            sample_weight: Sample weights
            
        Returns:
            Tuple of (best_feature_index, best_threshold, best_gain)
        """
        n_samples, n_features = X.shape
        
        # Calculate the weighted MSE before split
        weighted_sum = np.sum(y * sample_weight)
        total_weight = np.sum(sample_weight)
        if total_weight > 0:
            node_value = weighted_sum / total_weight
        else:
            node_value = 0.0
            
        initial_mse = np.sum(sample_weight * (y - node_value) ** 2) / total_weight if total_weight > 0 else 0.0
        
        best_feature = 0
        best_threshold = 0.0
        best_gain = -np.inf
        
        # Loop through all features
        for feature_idx in range(n_features):
            # Get unique values of the feature
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # If there's only one unique value, skip this feature
            if len(thresholds) <= 1:
                continue
            
            # For binary features, we can use a single threshold
            if len(thresholds) == 2:
                thresholds = [thresholds[0]]
            else:
                # For continuous features, use the midpoints between consecutive unique values
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            # Evaluate each threshold
            for threshold in thresholds:
                left_indices = feature_values <= threshold
                right_indices = ~left_indices
                
                # Skip if the split doesn't separate the data
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate weighted means for left and right splits
                left_weight = np.sum(sample_weight[left_indices])
                right_weight = np.sum(sample_weight[right_indices])
                
                if left_weight > 0:
                    left_mean = np.sum(y[left_indices] * sample_weight[left_indices]) / left_weight
                else:
                    left_mean = 0.0
                    
                if right_weight > 0:
                    right_mean = np.sum(y[right_indices] * sample_weight[right_indices]) / right_weight
                else:
                    right_mean = 0.0
                
                # Calculate weighted MSE after split
                left_mse = np.sum(sample_weight[left_indices] * (y[left_indices] - left_mean) ** 2) / left_weight if left_weight > 0 else 0.0
                right_mse = np.sum(sample_weight[right_indices] * (y[right_indices] - right_mean) ** 2) / right_weight if right_weight > 0 else 0.0
                
                # Calculate the weighted average MSE
                split_weight = (left_weight + right_weight)
                if split_weight > 0:
                    split_mse = (left_weight * left_mse + right_weight * right_mse) / split_weight
                else:
                    split_mse = 0.0
                
                # Calculate information gain
                gain = initial_mse - split_mse
                
                # Update the best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression values for samples in X.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        if self.root is None:
            raise ValueError("Tree not fitted yet")
        
        return np.array([self.root.predict(x) for x in X])


class GradientBoostingClassifier:
    """
    Gradient Boosting for classification.
    
    This implementation follows the algorithm described in Sections 10.9-10.10 of
    "The Elements of Statistical Learning" (2nd Edition) by Hastie, Tibshirani, and Friedman.
    """
    
    def __init__(self, 
                n_estimators: int = 100, 
                learning_rate: float = 0.1,
                max_depth: int = 3,
                min_samples_split: int = 2,
                min_impurity_decrease: float = 0.0,
                subsample: float = 1.0,
                random_state: Optional[int] = None):
        """
        Initialize the gradient boosting classifier.
        
        Args:
            n_estimators: Number of boosting stages (trees) to use
            learning_rate: Shrinkage parameter that scales the contribution of each tree
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum number of samples required to split a node
            min_impurity_decrease: Minimum decrease in impurity required for split
            subsample: Fraction of samples to use for fitting individual base learners
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample = subsample
        self.random_state = random_state
        
        self.trees = []
        self.initial_prediction = None
        self.classes_ = None
        
        # For binary classification
        self._rng = np.random.RandomState(self.random_state)
    
    def _to_binary_encoding(self, y: np.ndarray) -> np.ndarray:
        """
        Convert class labels to binary encoding.
        
        For binary classification, convert labels to {0, 1}.
        
        Args:
            y: Class labels
            
        Returns:
            Binary encoded labels
        """
        # Store unique classes
        self.classes_ = np.unique(y)
        
        if len(self.classes_) > 2:
            raise ValueError("This implementation only supports binary classification")
        
        # Map classes to 0, 1 encoding
        y_binary = np.zeros(len(y), dtype=np.int64)
        y_binary[y == self.classes_[1]] = 1
        
        return y_binary
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid function for binary classification.
        
        Args:
            x: Input values
            
        Returns:
            Sigmoid of input values, clipped to avoid numerical issues
        """
        # Clip to avoid overflow
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))
    
    def _negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the negative gradient of the loss function.
        
        For binary classification with log loss, this is y_true - sigmoid(y_pred).
        
        Args:
            y_true: True labels
            y_pred: Current predictions
            
        Returns:
            Negative gradient values
        """
        return y_true - self._sigmoid(y_pred)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """
        Build a gradient boosted classifier from the training set (X, y).
        
        Args:
            X: Training features
            y: Target class labels
            
        Returns:
            self
        """
        # Convert target to binary encoding if needed
        y_binary = self._to_binary_encoding(y)
        
        # Initialize prediction with log-odds of class proportions
        pos_count = np.sum(y_binary == 1)
        neg_count = len(y_binary) - pos_count
        
        # Initialize with log-odds ratio (add small constant to avoid division by zero)
        if pos_count > 0 and neg_count > 0:
            self.initial_prediction = np.log(pos_count / neg_count)
        else:
            self.initial_prediction = 0.0
            
        # Initialize predictions with initial values
        predictions = np.full(len(y_binary), self.initial_prediction)
        
        # Fit boosting stages (trees)
        for i in range(self.n_estimators):
            # Compute negative gradient (residuals)
            residuals = self._negative_gradient(y_binary, predictions)
            
            # Create a new regression tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease
            )
            
            # Sample data for this tree if subsample < 1
            if self.subsample < 1.0:
                sample_indices = self._rng.choice(
                    len(X), 
                    size=int(len(X) * self.subsample),
                    replace=False
                )
                X_subsample = X[sample_indices]
                residuals_subsample = residuals[sample_indices]
                
                # Fit tree on the sampled data
                tree.fit(X_subsample, residuals_subsample)
            else:
                # Fit tree on all data
                tree.fit(X, residuals)
            
            # Add the tree to our ensemble
            self.trees.append(tree)
            
            # Update predictions (apply learning rate)
            predictions += self.learning_rate * tree.predict(X)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X: Features to predict on
            
        Returns:
            Class probabilities (shape: [n_samples, n_classes])
        """
        if not self.trees:
            raise ValueError("Classifier not fitted yet")
        
        # Calculate raw predictions (log-odds)
        raw_predictions = self._raw_predict(X)
        
        # Apply sigmoid to get probabilities
        proba_positive = self._sigmoid(raw_predictions)
        
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.vstack([1 - proba_positive, proba_positive]).T
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted class labels
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        
        # Map back to original class labels
        return self.classes_[predictions]
    
    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw predictions (log odds) from the ensemble.
        
        Args:
            X: Features to predict on
            
        Returns:
            Raw predictions (log odds)
        """
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add up contributions from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the accuracy on the given test data and labels.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        return np.mean(self.predict(X) == y)
    
    def feature_importances(self) -> np.ndarray:
        """
        Compute feature importances based on reduction in MSE.
        
        This is a simplified version that assumes equal importance for each tree.
        
        Returns:
            Array of feature importances
        """
        if not self.trees:
            raise ValueError("Classifier not fitted yet")
        
        # Count the number of times each feature is used for splitting
        feature_counts = np.zeros(self.trees[0].n_features)
        
        def count_feature_usage(node):
            if node is None or node.is_leaf:
                return
            feature_counts[node.feature_index] += 1
            count_feature_usage(node.left)
            count_feature_usage(node.right)
        
        # Count feature usage in all trees
        for tree in self.trees:
            count_feature_usage(tree.root)
        
        # Normalize to get relative importance
        if np.sum(feature_counts) > 0:
            return feature_counts / np.sum(feature_counts)
        return feature_counts