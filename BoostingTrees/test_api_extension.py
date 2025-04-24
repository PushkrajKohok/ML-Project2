
import numpy as np
from model.GradientBoostingClassifier import GradientBoostingClassifier
#from model.GradientBoostingClassifier import GradientBoostingClassifier

clf = GradientBoostingClassifier(
    n_estimators=5,
    max_depth=3,
    learning_rate=0.5,
    min_samples_leaf=2,  #  This activates the fix!
    max_features=1
)

# Dummy binary classification dataset
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

# Convert labels to {-1, 1} as typically used in GBT
y_transformed = 2 * y - 1

# Instantiate and fit classifier
clf = GradientBoostingClassifier(n_estimators=5, max_depth=1, learning_rate=0.5)
clf.fit(X, y_transformed)

# Test predict_proba
proba = clf.predict_proba(X)
print("Probabilities:")
print(proba)

# Test score
score = clf.score(X, y_transformed)
print(f"Accuracy: {score:.2f}")

# Test staged_predict
print("Staged Predictions:")
for i, stage_pred in enumerate(clf.staged_predict(X)):
    print(f"Stage {i+1}: {stage_pred}")

print("Training loss per iteration:")
print(clf.errors_)

