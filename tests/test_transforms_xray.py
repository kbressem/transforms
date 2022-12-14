import pytest
import torch

# TODO
from trainlib.transforms import get_train_transforms
from trainlib.utils import load_config

from transforms.utils import REPO_DIR

CONFIG = load_config("configs/config_xray.yaml")

IMG_KEY = CONFIG.data.image_cols[0]
SEG_KEY = CONFIG.data.label_cols[0]

IMG_PATH = REPO_DIR / "tests/test_data/xray/Thorax_pa.dcm"
SEG_PATH = REPO_DIR / "tests/test_data/xray/Thorax_pa.seg.nrrd"


@pytest.mark.skip("Until trainlib solves https://github.com/kbressem/trainlib/issues/44")
def test_get_base_transforms():
    my_transforms = get_train_transforms(config=CONFIG)
    sample = {IMG_KEY: IMG_PATH, SEG_KEY: SEG_PATH}
    sample_transformed = my_transforms(sample)
    assert len(sample_transformed) == 4

    assert sample_transformed[IMG_KEY].shape == torch.Size([1, 2546, 2617])
    assert sample_transformed[SEG_KEY].shape == torch.Size([5, 2546, 2617])


if __name__ == "__main__":
    test_get_base_transforms()
