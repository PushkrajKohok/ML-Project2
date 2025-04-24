# Project 2

# Group Members:- 
## Fnu Hussain Bin Yousuf (A20580905)
## Pushkraj Kohok (A20592796)
## Rohit Lahori (A20582911)
## Rajni Pawar ()

---

## Boosting Trees

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

As before, please clone this repo, work on your solution as a fork, and then open a pull request to submit your assignment. *A pull request is required to submit and your project will not be graded without a PR.*

Put your README below. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?


## What does the model you have implemented do and when should it be used?

We have implemented a binary classification model using **Gradient Boosting Decision Trees**, trained from first principles with logistic loss. The model combines multiple shallow regression trees where each tree corrects the residuals (pseudo-gradients) of the previous ensemble. The model's predictions are additive in the function space and refined stage-by-stage using an optimized step size \( \gamma_m \), improving both accuracy and convergence.

This model is most effective in structured/tabular datasets, especially when feature relationships are non-linear and performance is improved by capturing high-order interactions. It is suitable for binary classification tasks where interpretability (via feature importances and loss monitoring) is also important.

---

## How did you test your model to determine if it is working reasonably correctly?

We employed a comprehensive and multi-layered testing strategy:

1. **Quantitative evaluation** on:
   - Synthetic datasets: moons, circles, linearly separable data
   - Metrics: accuracy, loss progression, gamma value tracking

2. **Visual diagnostics**:
   - Decision boundary plots
   - Training log-loss curves per iteration
   - Gamma value per boosting stage

3. **Structured testing suite** with over 30 test cases:
   - Functional tests (e.g., initialization, learning rate scaling, proba output)
   - Edge cases (NaN handling, all-labels-equal, repeated `.fit()`)
   - Stress and combination tests (deep trees + small leaves, high-dim with max_features)

These tests are located in the `tests/testBoostingTree.py` file and can be run using:
```bash
python -m unittest tests/testBoostingTree.py
```

---

## What parameters have you exposed to users of your implementation in order to tune performance?

Our implementation exposes the following hyperparameters:

- `n_estimators`: Number of trees to be trained in the boosting sequence
- `learning_rate`: Scaling factor on each new tree (shrinkage)
- `max_depth`: Maximum depth of individual decision trees
- `min_samples_leaf`: Minimum number of samples in a tree leaf node
- `max_features`: Number of features considered per split (feature subsampling)
- `subsample`: Fraction of training rows used per tree (data bagging)
- `verbose`: Boolean to control detailed printing/logs

### Example Usage:
```python
clf = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=2,
    max_features=1,
    subsample=0.8,
    verbose=False
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

---

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes, the model currently has the following known limitations:

1. **Multiclass classification** is not yet supported. The model is designed for binary classification only. Extending it would involve implementing softmax loss and one-vs-rest strategies.

2. **NaN or missing input values** are not automatically handled. This can be mitigated by integrating preprocessing modules such as `SimpleImputer` or by using surrogate splits.

3. **Highly imbalanced class distributions** may affect convergence in early stages, though our initialization using log-odds helps. Future work could include focal loss or cost-sensitive learning.

4. **Single-class labels (e.g., all y = 1)** cause the model to raise a `ValueError`, which is expected behavior. This is a fundamental requirement for supervised learning and cannot be bypassed meaningfully.





#  Gradient Boosting from Scratch 

This project involves the implementation of a binary classification algorithm based on **Gradient Boosting Decision Trees** using only core Python and NumPy. The model is optimized for log-loss and includes multiple advanced features such as optimized gamma calculation, custom decision trees with feature and data subsampling, staged prediction tracking, and thorough visualization for analysis and validation.

---

##  Overview

Gradient Boosting is an ensemble method that builds a sequence of weak learners (typically decision trees) where each new learner corrects the errors of the existing ensemble. Our implementation is focused on binary classification using the **logistic loss function** and supports the following core features:

- Initialization based on log-odds:
  $$ F_0(x) = \log\left(rac{p}{1 - p}
ight) $$
  where \( p \) is the proportion of positive samples.

- Computation of residuals (negative gradients):
  $$ r_i^{(m)} = y_i - \sigma(F_{m-1}(x_i)) $$

- Tree updates scaled by an optimal gamma:
  $$
  \gamma_m = rac{\sum_i (y_i - \sigma(F_{m-1}(x_i))) \cdot h_m(x_i)}{\sum_i \sigma(F_{m-1}(x_i)) (1 - \sigma(F_{m-1}(x_i))) \cdot h_m(x_i)^2}
  $$

- Log-loss convergence tracking across iterations
- Visualization of decision boundaries, gamma evolution, feature importances, and loss progression

The final model is highly configurable, robust, and extensively tested.

---


---

##  Project Objectives

- Implement a Gradient Boosting Classifier from scratch without external ML libraries.
- Support binary classification using the logistic loss function.
- Enable complete configurability via hyperparameters such as learning rate, max depth, subsample ratio, and more.
- Visualize internal model behavior including training loss, gamma scaling, and decision boundaries.
- Rigorously test all aspects of the model with both functional and corner-case test suites.

---

##  Configuration Parameters

The model supports the following key hyperparameters for tuning:

- `n_estimators` — Number of boosting rounds (trees).
- `learning_rate` — Multiplier applied to each tree's output.
- `max_depth` — Maximum depth of each regression tree.
- `min_samples_leaf` — Minimum number of samples per leaf node.
- `max_features` — Number of features considered per split.
- `subsample` — Fraction of training data used per tree.
- `verbose` — Toggles verbose training output.

---

##  Complete Test Case Inventory

###  Original Functional Tests
1. **test_initialization** – Tests basic instantiation of the model.
2. **test_linear_data** – Evaluates model accuracy on linearly separable data.
3. **test_nonlinear_data** – Tests accuracy on moons and circles datasets.
4. **test_gaussian_data** – Accuracy check on clustered Gaussian blobs.
5. **test_learning_rate** – Compares model accuracy across learning rates.
6. **test_n_estimators** – Measures accuracy impact of more trees.
7. **test_max_depth** – Analyzes tree depth influence on overfitting.
8. **test_subsample** – Tests effect of row-wise subsampling.
9. **test_predict_proba** – Confirms probabilities lie in [0, 1].
10. **test_visualization** – Generates and saves boundary plots.

###  Functional Extension Tests
11. **test_predict_shape** – Ensures `predict_proba()` has correct shape (n, 2).
12. **test_output_range_proba** – All probabilities are valid.
13. **test_loss_decreasing** – Log loss should decline monotonically.
14. **test_gamma_positivity** – Gamma values must be finite and > 0.
15. **test_feature_importance_sum** – Importance scores should sum to ~1.
16. **test_staged_predict_progression** – Staged predictions should converge.

###  Corner Case Tests
17. **test_large_n_estimators** – Tests model capacity with 500 trees.
18. **test_high_dimensional_data** – Supports 100+ features.
19. **test_single_feature** – Handles univariate inputs.
20. **test_duplicate_features** – Identical features shouldn’t confuse splits.
21. **test_class_imbalance** – Handles skewed class distributions.
22. **test_small_dataset_overfitting** – Overfits 5-sample dataset.
23. **test_float_class_labels** – Handles float labels like 0.0 and 1.0.
24. **test_single_class_label** – Raises error if all y are identical.
25. **test_repeated_fit** – Supports refitting on the same object.
26. **test_predict_before_fit** – Prevents prediction before training.
27. **test_nan_input_handling** – Raises error on NaN inputs.

###  Combination Stress Tests
28. **test_combo_deep_tree_small_leaf** – Overfitting: deep trees + small leaves.
29. **test_combo_shallow_tree_with_subsampling** – Underfitting: shallow + sampled data.
30. **test_combo_high_dim_low_max_features** – High-dim input + low `max_features`.
31. **test_combo_class_imbalance_loss_stability** – Monitors loss under imbalance.
32. **test_combo_gamma_vs_depth** – Tests gamma variability by depth.

---

##  Visual Analysis Components

Our `Final_notebook.ipynb` includes the following visualizations:

- Log-loss curves over iterations
- Gamma values per tree
- Decision boundaries (train/test)
- Feature importance bar plots
- Effect of hyperparameters like `min_samples_leaf` and `max_features`

---

##  Enhancements and Technical Strengths

###  Gamma Optimization:
Used analytic derivation for optimal scaling per iteration to ensure fast convergence and minimized loss.

###  Tree-based Feature Importance:
Calculated using gain-per-split, normalized:
\[
\text{importance}_j = \sum \text{gain}_{j} \quad \text{then normalized:} \sum_j \text{importance}_j = 1
\]

###  Sample and Feature Subsampling:
Implemented `subsample` and `max_features` for bias-variance tradeoffs and improved generalization.

###  Full Training Traceability:
Added `self.errors_`, `self.gammas`, `staged_predict()` — fully transparent model updates.

###  Modular Testing:
Structured for `pytest` compatibility with 32 granular test cases.

---

##  Future Work

- Add early stopping based on validation loss
- Handle missing values using split-aware imputers
- Extend to multiclass classification
- Implement pruning or tree dropout
- Export models as JSON or binary blobs

---

##  Troubleshooting Guide

| Symptom | Solution |
|---------|----------|
| Verbose output flooding console | Set `verbose=False` in model |
| Predicts same class always | Check if all `y` = same label |
| Error on NaNs | Use preprocessing to impute or drop missing rows |
| Crash during `predict()` | Ensure `.fit()` is called first |
| Low accuracy on small datasets | Tune `max_depth` and `min_samples_leaf` |

---
