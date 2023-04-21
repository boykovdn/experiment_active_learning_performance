from scipy.stats import entropy
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def select_query_subset(X, y, P, size, eps=1e-10):
    r"""
    Simulates selecting the query subset of X and receiving the 
    oracle predictions y for it. The subset is selected based on
    the highest entropy of the predicted distribution.
    
    Inputs:
        :X: (N, C) floats, feature vectors.

        :y: (N,) int, class belonging expect in {0,1} (Bernoulli).

        :P: (N, n_classes) Categorical distribution for each sample.

        :eps: float, epsilon to help avoid errors due to float 
            comparison.

    Returns:

        X, y sizes (size, C), (size,)
    """

    _entropies = np.apply_along_axis(entropy, 1, P) # (N,C) -> (N,)
    _value_limit = np.sort(_entropies)[size] - eps # boundary above which select.
    _subset_idx = _entropies > _value_limit # (N,) bool mask

    return X[_subset_idx], y[_subset_idx] 

def select_random_subset(X, y, size):
    
    assert size < len(X), "Subset ({}) cannot be larger than the dataset {}."\
            .format(size, len(X))

    _subset_idx = np.floor(
                np.random.uniform(0, len(X), size)
            ).astype(int)

    return X[_subset_idx], y[_subset_idx]

def gp_al(
        dset_train, 
        dset_eval, 
        classifier, 
        T=10, oracle_batch_size=20):
    r"""
    Simulates the training process of the classifier during the
    active learning loop. The ground truth is revealed gradually
    based on the entropy of the classifier output predictions.

    Inputs:

        :dset_train:

        :dset_eval:

        :classifier:

        :T: int

        :oracle_batch_size: int

    Returns:
        
        list of scores

    """
    X_train, y_train = dset_train
    X_eval, y_eval = dset_eval

    assert len(X_train) > oracle_batch_size,\
            "Dset len {} < batch size {}"\
            .format(len(X_train, oracle_batch_size))

    # Train classifier with a random subset of the data, of size equal to the
    # size which will be passed later to the oracle.
    X_init, y_init = select_random_subset(X_train, y_train, oracle_batch_size)
    classifier.fit(X_init, y_init) 
    scores = [classifier.score(X_eval, y_eval)]

    # Run the active learning loop.
    for t in range(T):

        P = classifier.predict_proba(X_train)
        X_subset, y_oracle = select_query_subset(X_train, y_train, P, oracle_batch_size)
        classifier.fit(X_subset, y_oracle)

        scores.append(classifier.score(X_eval, y_eval))

    return scores

def main():

    n_total=1000
    n_train=200
    T = 20
    oracle_batch_size = 10

    X, y = make_classification(random_state=1,
                           n_samples=n_total,
                           n_informative=20,
                           n_redundant=20,
                           n_repeated=20,
                           n_classes=2,
                           n_clusters_per_class=10,
                           n_features=60)

    X_train, y_train = X[:n_train], y[:n_train]
    X_eval, y_eval = X[n_train:], y[n_train:]

    clf = GaussianProcessClassifier()

    scores = gp_al(
            (X_train, y_train), 
            (X_eval, y_eval),
            clf,
            T=T,
            oracle_batch_size=oracle_batch_size
        )

    print(scores)
    # TODO Debug

if __name__ == "__main__":
    main()

