from cellpose.resnet_torch import CPnet
from torch.nn.functional import interpolate
import torch

class CPFeatures:
    
    def __init__(self, *args, pretrained_path=None, **kwargs):
        
        self.net = CPnet(*args, **kwargs)
        if pretrained_path is not None:
            self.net.load_model(pretrained_path)
        self.net.eval()
        
        self.hooked_outputs = {}
        self.register_hooks()
        
    def save_output_callback(self, layer_name, inp, out):
        self.hooked_outputs[layer_name] = out[0].detach()

    def hook_creator(self, layer_name):
        return (lambda layer_repr, inp, out : self.save_output_callback(layer_name, inp, out))
        
    def register_hooks(self):
        for mod_idx, mod_ in enumerate(self.net.modules()):
            if len(list(mod_.children())) == 0 and mod_._get_name() == "Conv2d":
                mod_.register_forward_hook(
                    self.hook_creator(mod_idx)
                )

    def hooked_to_array(self, square_size=128):
        r"""
        Restructures the dict of hooked intermediate outputs into a
        Tensor and upsamples the reduced spatial dimensions.

        Inputs:
            :square_size (optional): int, size to which to upsample,
                default to 128.

        Outputs:

            torch.Tensor (Cn, square_size, square_size)
        """
        outp_tensor = None

        assert self.hooked_outputs, "Hooked outputs are empty."
        for key, tensor_ in self.hooked_outputs.items():

            outp_ = interpolate(tensor_[None,], size=(square_size, square_size))[0]

            if outp_tensor is None:
                outp_tensor = outp_
            else:
                outp_tensor = torch.cat([outp_tensor, outp_], dim=0)

        return outp_tensor

    def __call__(self, x):
        r"""
        Inputs:

            :x: torch.Tensor [B,1,H,W] Grayscale image, the gray channel 
                is copied twice later. Not sure why, the CPnet expects this 
                format.

                Note: The batch dimension is computed via a for loop,
                      so no efficiency gains.

        Outputs:
            
            torch.Tensor [B, Cn, H, W] Feature encoding of the image.
                
        """
        assert len(x.shape) == 4, "Expected (B,C,H,W), got {}".format(x.shape)
        B,C,H,W = x.shape

        outp_ = []
        for b in range(B):
            _ = self.net(x[b,None].expand(-1,2,-1,-1))
            outp_.append(self.hooked_to_array())

        return torch.stack(outp_)
