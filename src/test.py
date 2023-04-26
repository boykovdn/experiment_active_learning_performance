from .models import CPFeatures
from .dataset import CellCrops
from .transforms import add_ellipse_random_placement

from pathlib import Path
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import torch

DSET_PATH = Path("/data/dataset/fl_dataset/healthy_train/raw")
dset = CellCrops(DSET_PATH, transforms=None, ext=None, load_to_gpu=None, set_size=128, debug=False)

cp = CPFeatures([2, 32, 64, 128, 256], 3, 3, diam_mean=70, pretrained_path="/model/cyto2torch_0")
#outp_ = cp((dset[0] / 255.)[None,].expand(-1,2,-1,-1))[0]
#print(outp_.shape)

img = dset[0]
img, mask = add_ellipse_random_placement(img, (60,70), (60,70), (10,20), (5,10), (0, 1.6))
print(img.shape)
print(mask.dtype)

def subsample(X, m):
    r"""
    Inputs:

        :X: (N_samples, N_features)

        :m: int

    Returns:
        (m, N_features)
    """
    N_samples = X.shape[0]
    sample_idxs = (torch.rand(m) * N_samples).floor().long()

    return X[sample_idxs]

def test_classifier_forward():
    r"""
    """

    def get_good_bad(img_ok, img_prebad):
        r"""
        Ret X,y balanced dataset.
        """

        img_bad, mask_bad = add_ellipse_random_placement(img_prebad, (60,70), (60,70), (10,20), (5,10), (0, 1.6))
        feats_bad = cp((img_bad / 255.)[None,].expand(-1,2,-1,-1))[0][:,mask_bad].T
        feats_ok = subsample(
                cp((img_ok / 255.)[None,].expand(-1,2,-1,-1))[0].flatten(start_dim=1).T,
                feats_bad.shape[0]
            )

        X = torch.cat([feats_ok, feats_bad], dim=0)
        y = torch.cat([torch.zeros(feats_ok.shape[0]),
                             torch.ones(feats_bad.shape[0])
                             ]).int()

        return X, y

    img_ok = dset[0]
    img_prebad = dset[1]
    X_train, y_train = get_good_bad(img_ok, img_prebad)

    img_ok_test = dset[2]
    img_prebad_test = dset[3]
    X_test, y_test = get_good_bad(img_ok_test, img_prebad_test)
    print(y_train.shape)
    print(y_test.shape)

    gp_clf = GaussianProcessClassifier(
                kernel=( ConstantKernel(0.5) * RBF(length_scale=2.) )
            )
    gp_clf.fit(X_train, y_train)
    print(gp_clf.kernel_)

    print(gp_clf.score(X_test, y_test))

test_classifier_forward()
