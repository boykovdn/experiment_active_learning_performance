from models import CPFeatures
from dataset import CellCrops
from pathlib import Path
import torch

DSET_PATH = Path("/data/dataset/fl_dataset/healthy_train/raw")
dset = CellCrops(DSET_PATH, transforms=None, ext=None, load_to_gpu=None, set_size=128, debug=False)

print(dset[0].shape)
cp = CPFeatures([2, 32, 64, 128, 256], 3, 3, diam_mean=70, pretrained_path="/model/cyto2torch_0")
outp_ = cp((dset[0] / 255.)[None,].expand(-1,2,-1,-1))[0]
print(outp_.shape)
