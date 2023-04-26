import os
import torch
import imageio
from tqdm.auto import tqdm
from torchvision.transforms import Resize

from sklearn.datasets import make_classification

from .utils import select_init_random_subset

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

def make_sklearn_generated_dataset(
        n_total=15000, n_train=10000,
        n_informative=2, n_redundant=0, 
        n_repeated=0, n_clusters_per_class=2,
        oracle_batch_size=10,
        random_state=0 
        ):
    r"""

    Inputs:

        :n_total: int, the total number of datapoints generated.

        :n_train: int, n_train < n_total. The number of datapoints used for
            training. The other n_total - n_train are used for score evaluation.

        :n_informative: int, the dimensionality of the Normal distribution from
            which the generated data is sampled. Each of these dimensions carries
            useful information.

        :n_redundant: int, linear combinations of the informative features, check
            sklearn make_classification.

        :n_repeated: int, number of repeated dimensions.

        :n_clusters_per_class: int, number of clusters each class has.

        :oracle_batch_size: int, the size of the batch queried from the oracle.
            This is needed because the classifiers are trained on the same
            initial subset and the function which selects these subsets ensures
            that a balanced selection of both classes is present, and needs to
            know how many points to sample.

        :random_state: int, sklearn seed for reproducibility.

    Returns:

        train, test, init datasets. Each one is a tuple of (X, y), where X is 
            of shape (N_samples, N_featues) and y is the binary classification
            label in {0, 1}.

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

    return (X_train, y_train), (X_eval, y_eval), (X_init, y_init)
