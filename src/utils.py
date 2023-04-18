from scipy.ndimage.morphology import distance_transform_edt
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
