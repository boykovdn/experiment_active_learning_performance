from scipy.stats import entropy
from sklearn.datasets import make_classification
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

def select_query_subset(X, y, P, size):
    r"""
    Simulates selecting the query subset of X and receiving the 
    oracle predictions y for it. The subset is selected based on
    the highest entropy of the predicted distribution.
    
    Inputs:
        :X: (N, C) floats, feature vectors.

        :y: (N,) int, class belonging expect in {0,1} (Bernoulli).

        :P: (N, n_classes) Categorical distribution for each sample.

    Returns:

        X, y sizes (size, C), (size,)
    """

    _entropies = np.apply_along_axis(entropy, 1, P) # (N,C) -> (N,)
    _idxs = np.argsort(_entropies)[-size:]

    return X[_idxs], y[_idxs] 

def select_random_subset(X, y, P, size):
    r"""
    P not used, but added for compatibility with experiment callable.
    """
    
    assert size < len(X), "Subset ({}) cannot be larger than the dataset {}."\
            .format(size, len(X))

    _subset_idx = np.floor(
                np.random.uniform(0, len(X), size)
            ).astype(int)

    return X[_subset_idx], y[_subset_idx]

def gp_al(
        dset_train, 
        dset_eval, 
        dset_initial,
        classifier, 
        subset_selector,
        T=10, oracle_batch_size=20):
    r"""
    Simulates the training process of the classifier during the
    active learning loop. The ground truth is revealed gradually
    based on the entropy of the classifier output predictions.

    Inputs:

        :dset_train:

        :dset_eval:

        :classifier:

        :subset_selector:

        :T: int

        :oracle_batch_size: int

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

def select_init_random_subset(X, y, size):
    r"""
    Random subset selection, but taking care not to select datapoints
    only belonging to one class. Select size//2 from each one.
    """
    _mask0 = y == 0
    X_class0 = X[_mask0]
    X_class1 = X[~_mask0]

    X_init_0, y_init_0 = select_random_subset(X_class0, np.zeros(len(X_class0)), None, size)
    X_init_1, y_init_1 = select_random_subset(X_class1, np.ones(len(X_class1)), None, size)

    out_ = (np.concatenate([X_init_0, X_init_1]), np.concatenate([y_init_0, y_init_1]))

    return out_

def binary_classification_scores(
        clf,
        n_total=15000, n_train=10000, 
        T=15, oracle_batch_size=10,
        n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=2,
        random_state=0):
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

        :n_total: int, the total number of datapoints generated.

        :n_train: int, n_train < n_total. The number of datapoints used for
            training. The other n_total - n_train are used for score evaluation.

        :T: int, the number of times the oracle is queried for classification
            labels.

        :oracle_batch_size: int, the number of datapoints the oracle is asked
            to label.

        :n_informative: int, the dimensionality of the Normal distribution from
            which the generated data is sampled. Each of these dimensions carries
            useful information.

        :n_redundant: int, linear combinations of the informative features, check 
            sklearn make_classification.

        :n_repeated: int, number of repeated dimensions.

        :n_clusters_per_class: int, number of clusters each class has.

        :random_state: int, sklearn seed for reproducibility. 

    Returns:
        
        list of float, active learning evaluation scores per iteration of T

        list of float, random learning
        
    """

    X, y = make_classification(
                           random_state=random_state,
                           n_samples=n_total,
                           n_informative=n_informative,
                           n_redundant=n_redundant,
                           n_repeated=n_repeated,
                           n_classes = 2, # Hardcode binary classification problem
                           n_clusters_per_class=n_clusters_per_class,
                           n_features=(n_informative + n_redundant + n_repeated))

    X_train, y_train = X[:n_train], y[:n_train]
    X_eval, y_eval = X[n_train:], y[n_train:]
    # Ensure that both classifiers start at the same place by fitting them
    # with the same initial random sample before their training schedules
    # diverge.
    X_init, y_init = select_init_random_subset(X_train, y_train, oracle_batch_size)

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

    kernel_constant = 1.
    kernel_lengthscale = 1.
    T = 15
    oracle_batch_size = 10

    n_experiment_repeats = 30
    n_informative_list = [3, 5, 7, 9, 11]
    n_clusters_per_class_list = [2, 3, 4, 5, 6]

    clf = GaussianProcessClassifier(
            kernel=( ConstantKernel(kernel_constant) * RBF(length_scale=kernel_lengthscale))
          )

    ######################## Experiment
    fig, axes = plt.subplots(len(n_informative_list), len(n_clusters_per_class_list))

    pbar = tqdm(total=(len(n_informative_list) * len(n_clusters_per_class_list)),
                desc="Drawing plots...")
    for h_idx, n_informative in enumerate(n_informative_list):
        for w_idx, n_clusters_per_class in enumerate(n_clusters_per_class_list):

            scores_query, scores_random = [], []

            for _ in range(n_experiment_repeats):
                qs, rs = binary_classification_scores(
                            clf,
                            n_informative=n_informative, 
                            n_clusters_per_class=n_clusters_per_class)
                scores_query.append(qs)
                scores_random.append(rs)

            for scores in [scores_query, scores_random]:
                stds = np.stack(scores).std(0)
                means = np.stack(scores).mean(0)
                n_dpts = np.arange(T + 1) * oracle_batch_size # T+1 due to initial data and scoring.

                axes[h_idx, w_idx].errorbar(n_dpts, means, yerr=stds, alpha=0.7)

            pbar.update()

    #######################
    plt.savefig("/test/test.svg")

if __name__ == "__main__":
    main()

