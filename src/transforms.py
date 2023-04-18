import torch
import numpy as np
from utils import get_ellipsoid_pattern

def add_ellipse_random_placement(
        image_tensor,
        h_range,
        w_range,
        a_range,
        b_range,
        angle_range,
        alpha=0.8):
    r"""
    Samples a randomly oriented and scaled ellipse artifact to the 
    image. The inputs are ranges for each parameter to be sampled
    from a uniform distribution.

    Inputs:

        :image_tensor: torch.Tensor (C,H,W)

        :h_range: tuple int (2,) h position of ellipse center.

        :w_range: tuple int (2,) w position of ellipse center.

        :a_range: tuple int (2,) major semi-axis

        :b_range: tuple int (2,) minor semi-axis

        :angle_range: tuple float (2,) in range(0,pi) ellipse rotation

        :alpha: float in range(0,1), blend parameter

    Outputs:
        
        torch.Tensor (C,H,W), image with pattern added, weighed by
            alpha value.

    """

    C,H,W = image_tensor.shape
    assert C == 1, "Expected grayscale (1,H,W), but got {}".format(image_tensor.shape)

    # Stack the input ranges for a,b and angle. Then sample uniformly from each range.
    ranges_ = torch.stack(list(map(lambda x : torch.Tensor(x), [h_range, w_range, a_range, b_range, angle_range]))) # (5,2)
    h, w, a, b, angle = (torch.rand(5) * (ranges_[:,1] - ranges_[:,0]) + ranges_[:,0]).floor()

    # Sample the ellipse given the center and geometry parameters.
    artefact_ = torch.from_numpy(get_ellipsoid_pattern((H,W), torch.Tensor([h,w]), a=a, b=b, angle=angle)).float()

    mask_ = artefact_.bool()
    # Calculate weighted sum of pixels where artefact is present.
    image_tensor[0, mask_] = image_tensor[0, mask_] * (1-alpha) + artefact_[mask_] * alpha

    return image_tensor, mask_
