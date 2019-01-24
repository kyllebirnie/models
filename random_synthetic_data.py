import numpy as np
from sklearn.datasets import make_classification


"""
Generating a random dataframe. The number of rows and columns is provided as input parameters to the component
"""

def __init__(self, engine):
    super(self.__class__, self).__init__(engine)

def _materialize(self, parent_data_objs, user_data):
    n_features = self._params.get('num_features', 21)
    n_samples = self._params.get('num_samples', 50)

    self._logger.info("PM: Configuration")
    self._logger.info("# Features:      [{}]".format(n_features))
    self._logger.info("# Samples:       [{}]".format(n_samples))

    # Create synthetic data using scikit learn
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=3, n_redundant=1,
                               n_classes=3, n_clusters_per_class=1, random_state=45)

    # Separate into features and labels
    features = X
    labels = y

    # Add noise to the data
    noisy_features = np.random.uniform(0, 10) * np.random.normal(0, 1, (n_samples, n_features))
    features = features + noisy_features

    self._logger.info("Generated random dataframe rows: {} cols: {})".format(n_samples, n_features))
    return [features, labels]
