from .dataset import make_sklearn_generated_dataset
from .utils import select_query_subset, select_random_subset

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

def gp_al(
        dset_train, 
        dset_eval, 
        dset_initial,
        classifier, 
        subset_selector,
        T=10, oracle_batch_size=20):
    r"""
    Simulates the training process of the classifier during the
    active learning loop. The ground truth is revealed gradually,
    using the subset_selector function. It is expected that this 
    function is used to select the next set of points with highest
    entropy of prediction, but it can be replaced with a random 
    selector for the purpose of experiment.

    It is expected that T * oracle_batch_size is relatively small,
    because the classifier should learn with few training examples.

    Inputs:

        :dset_train: tuple (X, y) training dataset.

        :dset_eval: tuple (X, y) evaluation dataset.

        :classifier: classifier object that supports .fit, .predict_proba,
            .score functions.

        :subset_selector: function which accepts a dataset, a classifier
            class prediction matrix, and an int representing the output
            subset size.

        :T: int, the number of times the oracle is queried with new data.
            This is also the number of times subset_selector is called.

        :oracle_batch_size: int, the size of the subset passed on to the
            oracle at each query.

    Returns:
        
        list of scores

    """
    X_train, y_train = dset_train
    X_eval, y_eval = dset_eval
    X_init, y_init = dset_initial

    assert len(X_train) > oracle_batch_size,\
            "Dset len {} < batch size {}"\
            .format(len(X_train, oracle_batch_size))

    # Train classifier with an initial subset of the data, of size equal to the
    # size which will be passed later to the oracle.
    classifier.fit(X_init, y_init) 
    scores = [classifier.score(X_eval, y_eval)]

    # Keep track of the training subset used to far.
    X_subset_train, y_subset_train = X_init, y_init

    # Run the active learning loop.
    for t in range(T):

        P = classifier.predict_proba(X_train)
        X_subset, y_oracle = subset_selector(X_train, y_train, P, oracle_batch_size)

        # Append the selected new training subset to the old.
        X_subset_train = np.concatenate((X_subset_train, X_subset), axis=0)
        y_subset_train = np.concatenate((y_subset_train, y_oracle), axis=0)
        classifier.fit(X_subset_train, y_subset_train)

        #classifier.fit(X_subset, y_oracle)

        scores.append(classifier.score(X_eval, y_eval))

    return scores

def binary_classification_scores(
        clf,
        datasets,
        T=15, 
        oracle_batch_size=10
    ):
    r"""
    Sets up and performs an active learning experiment in which a probabilistic
    classifier is trained using batches of datapoints of maximum entropy, and 
    also by using batches of random subsets. The aim is to contrast the learning
    speed between the two approaches. It is expected that the active learning
    querying method will increase the classifier performance faster and using
    fewer examples. 

    The experiment is a mock-up of a framework in which an oracle is periodically
    queried to provide labels for datapoints with the highest uncertainty in 
    prediction. Fewer queries is better.

    Inputs:

        :clf: Probabilistic classifier which supports fit and predict_proba methods
            as outlined in the sklearn API.

        :datasets: TODO

        :T: int, the number of times the oracle is queried for classification
            labels.

        :oracle_batch_size: int, the number of datapoints the oracle is asked
            to label.

    Returns:
        
        list of float, active learning evaluation scores per iteration of T

        list of float, random learning
        
    """

    # Unpack the datasets
    _train_dset, _eval_dset, _init_dset = datasets
    X_train, y_train = _train_dset
    X_eval, y_eval = _eval_dset
    X_init, y_init = _init_dset

    # Active querying
    scores_query = gp_al(
            (X_train, y_train), 
            (X_eval, y_eval),
            (X_init, y_init),
            clf,
            select_query_subset,
            T=T,
            oracle_batch_size=oracle_batch_size
        )

    # Random querying
    scores_random = gp_al(
            (X_train, y_train), 
            (X_eval, y_eval),
            (X_init, y_init),
            clf,
            select_random_subset,
            T=T,
            oracle_batch_size=oracle_batch_size
        )

    return scores_query, scores_random

def main():

    # GP parameters
    kernel_constant = 1.
    kernel_lengthscale = 1.

    # Experiment parameters
    T = 15
    oracle_batch_size = 10

    # Dataset parameters
    n_informative = 3
    n_clusters_per_class = 2
    random_state = 0

    clf = GaussianProcessClassifier(
            kernel=( 
                        ConstantKernel(kernel_constant) * \
                        RBF(length_scale=kernel_lengthscale)
            ))
    datasets = make_sklearn_generated_dataset(
            n_informative=n_informative,
            n_clusters_per_class=n_clusters_per_class,
            oracle_batch_size=oracle_batch_size,
            random_state=random_state
        )

    # Experiment
    queried_scores, random_scores = \
        binary_classification_scores(
            clf,
            datasets,
            T=T, 
            oracle_batch_size=oracle_batch_size
        )

    print("Queried scores:\n{}".format(queried_scores))
    print("Random scores:\n{}".format(random_scores))

if __name__ == "__main__":
    main()

