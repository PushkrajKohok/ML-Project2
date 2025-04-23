"""
Generate synthetic datasets for testing the GradientBoostingClassifier.

This script creates various synthetic datasets and saves them as CSV files
for use in testing and demonstrations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
import os

def create_directory_if_not_exists(directory):
    """Create the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_linear_dataset(n_samples=1000, test_size=0.2, random_state=42, filename='linear_data.csv'):
    """
    Generate a dataset with a linear decision boundary.
    
    Args:
        n_samples: Number of samples to generate
        test_size: Proportion of test samples
        random_state: Random seed for reproducibility
        filename: Output filename
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1
    )
    
    data = np.column_stack((X, y))
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
    
    return df

def generate_moons_dataset(n_samples=1000, noise=0.3, random_state=42, filename='moons_data.csv'):
    """
    Generate a dataset with two interleaving half moons.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        random_state: Random seed for reproducibility
        filename: Output filename
    """
    X, y = make_moons(
        n_samples=n_samples, 
        noise=noise, 
        random_state=random_state
    )
    
    data = np.column_stack((X, y))
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
    
    return df

def generate_circles_dataset(n_samples=1000, noise=0.2, factor=0.5, random_state=42, filename='circles_data.csv'):
    """
    Generate a dataset with two concentric circles.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise added to the data
        factor: Scale factor between inner and outer circle
        random_state: Random seed for reproducibility
        filename: Output filename
    """
    X, y = make_circles(
        n_samples=n_samples, 
        noise=noise, 
        factor=factor, 
        random_state=random_state
    )
    
    data = np.column_stack((X, y))
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
    
    return df

def generate_gaussian_dataset(n_samples=1000, random_state=42, filename='gaussian_data.csv'):
    """
    Generate a dataset with two Gaussian clusters.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        filename: Output filename
    """
    np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # Create two Gaussian clusters
    X1 = np.random.randn(n_per_class, 2) + np.array([2, 2])  # Class 0
    X2 = np.random.randn(n_per_class, 2) + np.array([-2, -2])  # Class 1
    
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    data = np.column_stack((X, y))
    df = pd.DataFrame(data, columns=['feature1', 'feature2', 'label'])
    
    return df

def generate_complex_dataset(n_samples=1000, n_features=10, random_state=42, filename='complex_data.csv'):
    """
    Generate a more complex dataset with more features and nonlinear relationships.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate initially
        random_state: Random seed for reproducibility
        filename: Output filename
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=3,
        class_sep=1.0,
        flip_y=0.1,
        random_state=random_state
    )
    
    # Add some nonlinear interactions
    X_nonlinear = np.zeros((n_samples, n_features + 3))
    X_nonlinear[:, :n_features] = X
    X_nonlinear[:, n_features] = X[:, 0] * X[:, 1]  # Interaction term
    X_nonlinear[:, n_features+1] = X[:, 0]**2  # Quadratic term
    X_nonlinear[:, n_features+2] = np.sin(X[:, 0])  # Nonlinear transformation
    
    # Create column names
    columns = [f'feature{i+1}' for i in range(n_features)]
    columns.extend(['interaction', 'quadratic', 'sine'])
    columns.append('label')
    
    data = np.column_stack((X_nonlinear, y))
    df = pd.DataFrame(data, columns=columns)
    
    return df

def generate_all_datasets(output_dir='tests/data'):
    """
    Generate all datasets and save them to CSV files.
    
    Args:
        output_dir: Directory to save the datasets
    """
    create_directory_if_not_exists(output_dir)
    
    # Generate datasets
    linear_df = generate_linear_dataset()
    moons_df = generate_moons_dataset()
    circles_df = generate_circles_dataset()
    gaussian_df = generate_gaussian_dataset()
    complex_df = generate_complex_dataset()
    
    # Save datasets
    linear_df.to_csv(f'{output_dir}/linear_data.csv', index=False)
    moons_df.to_csv(f'{output_dir}/moons_data.csv', index=False)
    circles_df.to_csv(f'{output_dir}/circles_data.csv', index=False)
    gaussian_df.to_csv(f'{output_dir}/gaussian_data.csv', index=False)
    complex_df.to_csv(f'{output_dir}/complex_data.csv', index=False)
    
    print(f"Datasets successfully generated and saved to {output_dir}:")
    print(f"  - Linear dataset: {output_dir}/linear_data.csv")
    print(f"  - Moons dataset: {output_dir}/moons_data.csv")
    print(f"  - Circles dataset: {output_dir}/circles_data.csv")
    print(f"  - Gaussian dataset: {output_dir}/gaussian_data.csv")
    print(f"  - Complex dataset: {output_dir}/complex_data.csv")

if __name__ == "__main__":
    generate_all_datasets()