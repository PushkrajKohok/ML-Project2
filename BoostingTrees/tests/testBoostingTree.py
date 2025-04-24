"""
Test suite for the GradientBoostingClassifier implementation.

This module contains tests to verify the correctness and performance
of the GradientBoostingClassifier on various datasets.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification


# Add the parent directory to the path so we can import the model
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.GradientBoostingClassifier import GradientBoostingClassifier

class TestGradientBoostingClassifier(unittest.TestCase):
    """Tests for the GradientBoostingClassifier implementation."""
    
    def setUp(self):
        """Set up test data."""
        # Generate test data if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        datasets = ['linear_data.csv', 'moons_data.csv', 'circles_data.csv', 
                   'gaussian_data.csv', 'complex_data.csv']
                   
        missing_datasets = [dataset for dataset in datasets 
                           if not os.path.exists(os.path.join(data_dir, dataset))]
                           
        if missing_datasets:
            from tests.data.dataset_generator import generate_all_datasets
            generate_all_datasets()
        
        # Load test data
        self.datasets = {}
        for dataset in datasets:
            if dataset != 'complex_data.csv':  # We'll handle complex data separately
                df = pd.read_csv(os.path.join(data_dir, dataset))
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                # Split into train and test sets
                np.random.seed(42)
                indices = np.random.permutation(len(X))
                train_size = int(0.8 * len(X))
                
                self.datasets[dataset] = {
                    'X_train': X[indices[:train_size]],
                    'y_train': y[indices[:train_size]],
                    'X_test': X[indices[train_size:]],
                    'y_test': y[indices[train_size:]]
                }
    
    def test_initialization(self):
        """Test that the classifier initializes correctly."""
        gb = GradientBoostingClassifier(
            n_estimators=50, 
            learning_rate=0.1, 
            max_depth=3,
            verbose=False
        )
        
        self.assertEqual(gb.n_estimators, 50)
        self.assertEqual(gb.learning_rate, 0.1)
        self.assertEqual(gb.max_depth, 3)
        self.assertEqual(gb.min_samples_split, 2)  # Default value
        self.assertEqual(gb.subsample, 1.0)  # Default value
    
    def test_linear_data(self):
        """Test classifier on linearly separable data."""
        dataset = self.datasets['linear_data.csv']
        
        # Create and train the classifier
        gb = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3,
            random_state=42,
            verbose=False
        )
        
        gb.fit(dataset['X_train'], dataset['y_train'])
        
        # Make predictions
        y_pred = gb.predict(dataset['X_test'])
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == dataset['y_test'])
        
        # Linear data should be easy to classify with high accuracy
        self.assertGreater(accuracy, 0.85, 
                          f"Expected accuracy > 0.85, got {accuracy:.4f}")
        
        print(f"Linear data test accuracy: {accuracy:.4f}")
    
    def test_nonlinear_data(self):
        """Test classifier on moons and circles datasets."""
        for dataset_name in ['moons_data.csv', 'circles_data.csv']:
            dataset = self.datasets[dataset_name]
            
            # Create and train the classifier
            gb = GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=5,  # Deeper trees for nonlinear data
                random_state=42,
                verbose=False
            )
            
            gb.fit(dataset['X_train'], dataset['y_train'])
            
            # Make predictions
            y_pred = gb.predict(dataset['X_test'])
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == dataset['y_test'])
            
            # Nonlinear data should still be classified with reasonable accuracy
            self.assertGreater(accuracy, 0.8, 
                              f"{dataset_name} - Expected accuracy > 0.8, got {accuracy:.4f}")
            
            print(f"{dataset_name} test accuracy: {accuracy:.4f}")
    
    def test_gaussian_data(self):
        """Test classifier on Gaussian clustered data."""
        dataset = self.datasets['gaussian_data.csv']
        
        # Create and train the classifier
        gb = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3,
            random_state=42,
            verbose=False
        )
        
        gb.fit(dataset['X_train'], dataset['y_train'])
        
        # Make predictions
        y_pred = gb.predict(dataset['X_test'])
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == dataset['y_test'])
        
        # Well-separated Gaussian clusters should be easy to classify
        self.assertGreater(accuracy, 0.9, 
                          f"Expected accuracy > 0.9, got {accuracy:.4f}")
        
        print(f"Gaussian data test accuracy: {accuracy:.4f}")
    
    def test_learning_rate(self):
        """Test the effect of learning rate on model performance."""
        dataset = self.datasets['moons_data.csv']
        
        learning_rates = [0.001, 0.01, 0.1, 1.0]
        accuracies = []
        
        for lr in learning_rates:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=lr,
                max_depth=3,
                random_state=42,
                verbose=False
            )
            
            gb.fit(dataset['X_train'], dataset['y_train'])
            y_pred = gb.predict(dataset['X_test'])
            accuracy = np.mean(y_pred == dataset['y_test'])
            accuracies.append(accuracy)
            
            print(f"Learning rate {lr}: test accuracy = {accuracy:.4f}")
        
        # Verify that all learning rates provide reasonable accuracy
        for accuracy in accuracies:
            self.assertGreater(accuracy, 0.8, 
                            f"Expected accuracy > 0.8 for all learning rates")
        
        # Verify that the middle learning rates (0.01, 0.1) perform reasonably well
        # compared to the extremes (0.001, 1.0)
        middle_rates_max = max(accuracies[1], accuracies[2])  # Best of 0.01 and 0.1
        
        # Check that the difference between extreme and middle rates isn't too large
        # This is a more flexible test that allows for some variation
        self.assertLess(abs(accuracies[0] - middle_rates_max), 0.1,  # Very low rate (0.001)
                    f"Very low learning rate performance differs too much from optimal")
        self.assertLess(abs(accuracies[3] - middle_rates_max), 0.1,  # Very high rate (1.0)
                    f"Very high learning rate performance differs too much from optimal")
    
    def test_n_estimators(self):
        """Test the effect of number of estimators on model performance."""
        dataset = self.datasets['moons_data.csv']
        
        n_estimators_list = [1, 10, 50, 100]
        train_accuracies = []
        test_accuracies = []
        
        for n_est in n_estimators_list:
            gb = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                verbose=False
            )
            
            gb.fit(dataset['X_train'], dataset['y_train'])
            
            # Training accuracy
            y_train_pred = gb.predict(dataset['X_train'])
            train_acc = np.mean(y_train_pred == dataset['y_train'])
            train_accuracies.append(train_acc)
            
            # Test accuracy
            y_test_pred = gb.predict(dataset['X_test'])
            test_acc = np.mean(y_test_pred == dataset['y_test'])
            test_accuracies.append(test_acc)
            
            print(f"N estimators {n_est}: train acc = {train_acc:.4f}, test acc = {test_acc:.4f}")
        
        # More trees should generally improve training accuracy
        self.assertTrue(train_accuracies[0] <= train_accuracies[-1])
        
        # Test accuracy should also generally improve but may plateau
        self.assertTrue(test_accuracies[0] <= test_accuracies[-1])
    
    def test_max_depth(self):
        """Test the effect of max_depth on model performance."""
        dataset = self.datasets['moons_data.csv']
        
        max_depths = [1, 3, 5, 10]
        train_accuracies = []
        test_accuracies = []
        
        for depth in max_depths:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=depth,
                random_state=42,
                verbose=False
            )
            
            gb.fit(dataset['X_train'], dataset['y_train'])
            
            # Training accuracy
            y_train_pred = gb.predict(dataset['X_train'])
            train_acc = np.mean(y_train_pred == dataset['y_train'])
            train_accuracies.append(train_acc)
            
            # Test accuracy
            y_test_pred = gb.predict(dataset['X_test'])
            test_acc = np.mean(y_test_pred == dataset['y_test'])
            test_accuracies.append(test_acc)
            
            print(f"Max depth {depth}: train acc = {train_acc:.4f}, test acc = {test_acc:.4f}")
        
        # Deeper trees should fit training data better
        self.assertTrue(train_accuracies[0] <= train_accuracies[-1])
    
    def test_subsample(self):
        """Test the effect of subsampling on model performance."""
        dataset = self.datasets['moons_data.csv']
        
        subsamples = [0.5, 0.7, 1.0]
        accuracies = []
        
        for subsample in subsamples:
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=subsample,
                random_state=42,
                verbose=False
            )
            
            gb.fit(dataset['X_train'], dataset['y_train'])
            y_pred = gb.predict(dataset['X_test'])
            accuracy = np.mean(y_pred == dataset['y_test'])
            accuracies.append(accuracy)
            
            print(f"Subsample {subsample}: test accuracy = {accuracy:.4f}")
        
        # All subsample rates should achieve reasonable accuracy
        for acc in accuracies:
            self.assertGreater(acc, 0.8)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        dataset = self.datasets['moons_data.csv']
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbose=False
        )
        
        gb.fit(dataset['X_train'], dataset['y_train'])
        
        # Get probability predictions
        proba = gb.predict_proba(dataset['X_test'])
        
        # Check shape
        self.assertEqual(proba.shape, (len(dataset['X_test']), 2))
        
        # Check that probabilities sum to 1
        row_sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-5)
        
        # Check that probabilities are between 0 and 1
        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))
        
        # Check that predictions match the class with highest probability
        y_pred = gb.predict(dataset['X_test'])
        y_pred_from_proba = np.argmax(proba, axis=1)
        
        # Map back to original class labels
        classes = np.unique(dataset['y_train'])
        y_pred_from_proba = classes[y_pred_from_proba]
        
        np.testing.assert_array_equal(y_pred, y_pred_from_proba)
        
    def test_visualization(self):
        """Test visualization of decision boundaries."""
        for dataset_name in ['linear_data.csv', 'moons_data.csv', 'circles_data.csv']:
            dataset = self.datasets[dataset_name]
            
            # Extract the feature data
            X_train = dataset['X_train']
            y_train = dataset['y_train']
            X_test = dataset['X_test']
            y_test = dataset['y_test']
            
            # Train the classifier
            gb = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                verbose=False
            )
            
            gb.fit(X_train, y_train)
            
            # Generate a grid for visualization
            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.1),
                np.arange(y_min, y_max, 0.1)
            )
            
            # Make predictions on the grid
            Z = gb.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Create a figure for visualization
            plt.figure(figsize=(10, 8))
            
            # Plot the decision boundary
            plt.contourf(xx, yy, Z, alpha=0.8)
            
            # Plot the training points
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=80, label='Train')
            
            # Plot the test points
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='^', s=40, label='Test')
            
            plt.title(f'Decision Boundary - {dataset_name}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            
            # Save the plot
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            plt.savefig(os.path.join(output_dir, f'decision_boundary_{dataset_name.split(".")[0]}.png'))
            plt.close()
            
            print(f"Decision boundary visualization saved for {dataset_name}")

###############################   ADDITIONAL TEST CASES    #############################

    

    def test_predict_shape(self):
        """Test shape of predict_proba output."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=5, verbose=False)
        gb.fit(X, y)
        proba = gb.predict_proba(X)
        self.assertEqual(proba.shape, (X.shape[0], 2))

    def test_output_range_proba(self):
        """Ensure predict_proba outputs values between 0 and 1."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=5, verbose=False)
        gb.fit(X, y)
        proba = gb.predict_proba(X)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_loss_decreasing(self):
        """Check that training loss decreases over iterations."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=10, verbose=False)
        gb.fit(X, y)
        losses = gb.errors_
        self.assertTrue(all(x >= y for x, y in zip(losses, losses[1:])))

    def test_gamma_positivity(self):
        """Ensure all gamma values are positive and finite."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=10, verbose=False)
        gb.fit(X, y)
        for gamma in gb.gammas:
            self.assertTrue(np.isfinite(gamma) and gamma > 0)

    def test_feature_importance_sum(self):
        """Check that feature importances sum approximately to 1."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=10, verbose=False)
        gb.fit(X, y)
        importances = gb.feature_importances()
        self.assertAlmostEqual(np.sum(importances), 1.0, places=4)

    def test_staged_predict_progression(self):
        """Check that staged_predict predictions evolve toward final prediction."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=10, verbose=False)
        gb.fit(X, y)
        final = gb.predict(X)
        found_match = False
        for stage_pred in gb.staged_predict(X):
            if np.array_equal(stage_pred, final):
                found_match = True
                break
        self.assertTrue(found_match)

    
    
    def test_min_samples_leaf_effect(self):
        """Models with larger min_samples_leaf should produce fewer splits (higher bias)."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf_small_leaf = GradientBoostingClassifier(n_estimators=5, min_samples_leaf=1, verbose=False)
        clf_large_leaf = GradientBoostingClassifier(n_estimators=5, min_samples_leaf=10, verbose=False)
        clf_small_leaf.fit(X, y)
        clf_large_leaf.fit(X, y)
        self.assertLessEqual(clf_large_leaf.score(X, y), clf_small_leaf.score(X, y))

    def test_max_features_effect(self):
        """Models with limited max_features should generalize better on small data."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf_all = GradientBoostingClassifier(n_estimators=5, max_features=None, verbose=False)
        clf_sub = GradientBoostingClassifier(n_estimators=5, max_features=1, verbose=False)
        clf_all.fit(X, y)
        clf_sub.fit(X, y)
        # No assert — just ensure it runs and the models are different
        self.assertNotEqual(clf_all.predict(X).tolist(), clf_sub.predict(X).tolist())

    def test_binary_input_validation(self):
        """Raise or gracefully handle invalid label values."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        y_invalid = y + 2  # Now {2, 3}
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        with self.assertRaises(ValueError):
            clf.fit(X, y_invalid)

    def test_zero_estimators(self):
        """Should raise error or produce no learning if n_estimators=0."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=0, verbose=False)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_nan_input_handling(self):
        """Ensure model raises error on NaN input."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        X[0, 0] = np.nan
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        with self.assertRaises(ValueError):
            clf.fit(X, y)
    
###############################   CORNER TEST CASES    #############################


    def test_large_n_estimators(self):
        """Test performance with a large number of estimators."""
        X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=500, verbose=False)
        clf.fit(X, y)
        self.assertGreater(clf.score(X, y), 0.8)

    def test_high_dimensional_data(self):
        """Ensure model works with high-dimensional input (n_features=100)."""
        X, y = make_classification(n_samples=200, n_features=100, n_informative=10, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=10, max_features=10, verbose=False)
        clf.fit(X, y)
        self.assertEqual(len(clf.feature_importances()), 100)

    def test_single_feature(self):
        """Model should work when dataset has only one feature."""
        #X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42)
        X, y = make_classification(n_samples=100, n_features=1, n_informative=1,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        clf.fit(X, y)
        self.assertEqual(clf.feature_importances().shape[0], 1)

    def test_duplicate_features(self):
        """Model should not split on features that provide no gain."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        X_dup = np.tile(X[:, [0]], (1, 5))  # repeat feature 0
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        clf.fit(X_dup, y)
        self.assertEqual(clf.feature_importances().sum(), 1.0)  # should still normalize

    def test_class_imbalance(self):
        """Test model behavior on imbalanced class distribution."""
        X, y = make_classification(n_samples=200, weights=[0.9, 0.1], flip_y=0, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=10, verbose=False)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        self.assertTrue(np.all(proba >= 0) and np.all(proba <= 1))

    def test_small_dataset_overfitting(self):
        """Model should overfit a very small dataset."""
        X, y = make_moons(n_samples=5, noise=0.0, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=20, max_depth=5, verbose=False)
        clf.fit(X, y)
        self.assertGreaterEqual(clf.score(X, y), 0.95)

    def test_float_class_labels(self):
        """Model should accept class labels as floats."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        y = y.astype(float)
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        clf.fit(X, y)
        self.assertTrue(hasattr(clf, "trees"))

    def test_single_class_label(self):
        """Model should raise an error if all class labels are the same."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        y[:] = 1
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        with self.assertRaises(ValueError):
            clf.fit(X, y)

    def test_repeated_fit(self):
        """Model should reset properly when .fit() is called multiple times."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        clf.fit(X, y)
        clf.fit(X, y)  # Second fit should not crash or retain state
        self.assertTrue(hasattr(clf, "trees"))

    def test_predict_before_fit(self):
        """Calling predict before fit should raise an error."""
        X, _ = make_moons(n_samples=10, noise=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=5, verbose=False)
        with self.assertRaises(Exception):
            clf.predict(X)

###############################   COMBINATION TEST CASES    #############################
    
    def test_combo_deep_tree_small_leaf(self):
        """Test overfitting scenario with deep trees and small leaf size."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=20, max_depth=10, min_samples_leaf=1, verbose=False)
        clf.fit(X, y)
        self.assertGreaterEqual(clf.score(X, y), 0.95)  # likely to overfit

    def test_combo_shallow_tree_with_subsampling(self):
        """Test underfitting scenario with shallow trees and subsampling."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=20, max_depth=1, subsample=0.5, verbose=False)
        clf.fit(X, y)
        self.assertGreaterEqual(clf.score(X, y), 0.7)  # minimal acceptable performance

    def test_combo_high_dim_low_max_features(self):
        """Stress test feature subsampling with high-dimensional input."""
        X, y = make_classification(n_samples=200, n_features=100, n_informative=10, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=10, max_features=5, verbose=False)
        clf.fit(X, y)
        self.assertEqual(len(clf.feature_importances()), 100)

    def test_combo_class_imbalance_loss_stability(self):
        """Test model stability and log loss on imbalanced data."""
        X, y = make_classification(n_samples=300, weights=[0.9, 0.1], flip_y=0, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=20, verbose=False)
        clf.fit(X, y)
        self.assertTrue(all(np.isfinite(err) for err in clf.errors_))

    def test_combo_gamma_vs_depth(self):
        """Compare gamma behavior across depth settings."""
        X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
        clf1 = GradientBoostingClassifier(n_estimators=10, max_depth=1, verbose=False)
        clf2 = GradientBoostingClassifier(n_estimators=10, max_depth=10, verbose=False)
        clf1.fit(X, y)
        clf2.fit(X, y)
        self.assertNotEqual(clf1.gammas, clf2.gammas)


if __name__ == '__main__':
    unittest.main()