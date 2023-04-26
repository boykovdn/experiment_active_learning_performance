from scipy.ndimage.morphology import distance_transform_edt
from scipy.stats import entropy

import numpy as np
import torch

def rescale_to(img, to=(0., 255.), eps_=1e-6):
    r"""
    :img: [B,C,*]
    """
    outp_min = to[0]
    outp_max = to[1]

    outp_ = img - img.min()
    outp_ = outp_ / (outp_.max() + eps_)
    outp_ = outp_ * (outp_max - outp_min)
    outp_ = outp_ + outp_min

    return outp_

def get_ellipsoid_mask(out_shape, center, a=10., b=5., angle=0.3):
    cos_ = torch.cos(torch.Tensor([angle])).item()
    sin_ = torch.sin(torch.Tensor([angle])).item()
    rot_ = torch.Tensor([[cos_, -sin_],
            [sin_, cos_]]) # (2,2)

    mask = torch.tensor(np.indices(out_shape))
    dist_ = (mask - center[:,None,None])
    dist_ = torch.einsum("ij,jkl->ikl", rot_,dist_)
    dist_[0] /= a
    dist_[1] /= b
    dist_ = dist_.square().sum(0).sqrt()

    mask = (dist_ < 1).bool()

    return mask

def get_ellipsoid_pattern(out_shape, center, a=10., b=5., angle=0.3):
    r"""
    """
    mask = get_ellipsoid_mask(out_shape, center, a=a, b=b, angle=angle)
    dt = distance_transform_edt(mask.numpy())
    pattern = rescale_to(dt, to=(0,1))

    return pattern

def get_ellipsoid_noise(out_shape, center, a=10., b=5., angle=0.3, 
        stdev=0.5):
    r"""
    """
    mask = get_ellipsoid_mask(out_shape, center, a=a, b=b, angle=angle)
    random_sample = torch.randn(mask.sum()) * stdev
    pattern = torch.zeros(*out_shape)
    pattern[mask] = random_sample

    return pattern

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

