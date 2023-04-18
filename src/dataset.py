import os
import torch
import imageio
from tqdm.auto import tqdm
from torchvision.transforms import Resize

class CellCrops(torch.utils.data.Dataset):
    r"""
    Loads all cells into memory, they should be very small - cropped around
    RBCs. There is no ground truth, so we only load the raw images (used for
    autoenoder training).
    """
    
    def __init__(self, img_path, transforms=None, ext=None, load_to_gpu=0, set_size=128, debug=False):
        r"""
        Args:
            :img_path: str/path. Points to the folder containing the images.
        """
        self.transforms = transforms

        self.set_size = set_size
        self.img_size = (set_size, set_size)
        self.cell_names = os.listdir(img_path)
        if isinstance(debug, bool):
            self.debug = debug
            self._debug_len = 500
        else:
            self.debug = True
            self._debug_len = debug

        if debug:
            self.dset_len = self._debug_len
        else:
            self.dset_len = len(self.cell_names)

        if ext is not None:
            self.cell_names = [name for name in cell_names if ext in name]

        # Sorting is not necessary, but might aid debugging later.
        self.cell_names.sort()

        if debug:
            self.cell_names = self.cell_names[:self._debug_len]

        self.cell_images = self._load_cells_from_names(img_path, self.cell_names)
        if load_to_gpu is not None:
            self.cell_images = self.cell_images.to()
        self.img_shape = self._get_image_shape()

    def __len__(self):
        return self.dset_len

    def _get_image_shape(self):
        return self.cell_images[0].shape

    def _load_cells_from_names(self, img_path, cell_names):
        r"""
        Load the images as np arrays, then turns them into float32.
        """
        temp_img = imageio.imread("{}/{}".format(img_path, cell_names[0]))
        unsq = False
        if len(temp_img.shape) == 2:
            # Image is grayscale, so no channel is loaded.
            outp_shape = (1, *self.img_size)
            unsq = True
        elif len(temp_img.shape) == 3:
            # Expect a RGB image, channels first
            outp_shape = (3, *self.img_size)
        else:
            raise Exception("Loaded unsupported img size. Expected Gray or RGB (channels first)")

        if self.debug:
            temp_array = torch.zeros((self._debug_len, *outp_shape))
        else:
            temp_array = torch.zeros((self.__len__(), *outp_shape)) # (N,C,*)
        resize = Resize(self.img_size)

        if not unsq:
            for idx, img_name in enumerate(tqdm(self.cell_names, desc="Loading crops...")):
                temp_array[idx] = resize(
                        torch.from_numpy(
                            imageio.imread(
                                "{}/{}".format(img_path, img_name))).float().transpose(0,-1)
                        )
        if unsq:
            for idx, img_name in enumerate(tqdm(self.cell_names, desc="Loading crops...")):
                temp_array[idx] = resize(
                        torch.from_numpy(
                            imageio.imread(
                                "{}/{}".format(img_path, img_name))).float().transpose(0,-1).unsqueeze(0)
                        )

        return temp_array

    def __getitem__(self, idx):
        r"""
        Args:
            :idx: int
        Returns:
            [C,*], float32 image array.
        """
        if self.transforms is None:
            return self.cell_images[idx]
        else:
            return self.transforms(self.cell_images[idx])
