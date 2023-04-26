import torch
import numpy as np
from .utils import get_ellipsoid_pattern

def add_ellipse_random_placement(
        image_tensor,
        h_range,
        w_range,
        a_range,
        b_range,
        angle_range,
        scale=1.):
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

        :scale: float, the intensity of the artefact.

    Outputs:
        
        torch.Tensor (C,H,W), image with pattern added, weighed by
            alpha value.

    """

    C,H,W = image_tensor.shape
    assert C == 1, "Expected grayscale (1,H,W), but got {}".format(image_tensor.shape)

    # Stack the input ranges for a,b and angle. Then sample uniformly from each range.
    ranges_ = torch.stack(list(map(lambda x : torch.Tensor(x), [h_range, w_range, a_range, b_range]))) # (5,2)
    h, w, a, b = (torch.rand(4) * (ranges_[:,1] - ranges_[:,0]) + ranges_[:,0]).floor()
    angle = torch.rand(1) * (angle_range[1] - angle_range[0]) + angle_range[0]

    # Sample the ellipse given the center and geometry parameters.
    artefact_ = torch.from_numpy(get_ellipsoid_pattern((H,W), torch.Tensor([h,w]), a=a, b=b, angle=angle)).float()
    
    mask_ = artefact_.bool()
    # Add ellipse artifact.
    image_tensor[0, mask_] = image_tensor[0, mask_] + artefact_[mask_] * scale

    return image_tensor, mask_
