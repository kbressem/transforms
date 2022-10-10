import torch
from trainlib.transforms import get_train_transforms
from trainlib.utils import load_config

from deepspa.utils import REPO_DIR

CONFIG = load_config("configs/config_mri.yaml")

IMG_KEY = CONFIG.data.image_cols[0]
SEG_KEY = CONFIG.data.label_cols[0]

IMG_PATH = REPO_DIR / "tests/test_data/mri/K_B_Seq_2_t2_tse_sag"
SEG_PATH = REPO_DIR / "tests/test_data/mri/K_B_Seq_2_t2_tse_sag.seg.nrrd"


def test_get_base_transforms():
    my_transforms = get_train_transforms(config=CONFIG)
    sample = {IMG_KEY: IMG_PATH, SEG_KEY: SEG_PATH}
    sample_transformed = my_transforms(sample)
    print(sample_transformed.keys())
    print("shape", sample_transformed[IMG_KEY].shape, sample_transformed[SEG_KEY].shape)
    assert len(sample_transformed) == 4
    assert sample_transformed[IMG_KEY].shape == torch.Size([1, 521, 535])


def test_get_base_transforms():
    my_transforms = get_train_transforms(config=CONFIG)
    sample = {IMG_KEY: IMG_PATH, SEG_KEY: SEG_PATH}
    sample_transformed = my_transforms(sample)

    assert len(sample_transformed) == 4
    assert sample_transformed[IMG_KEY].shape == torch.Size([1, 384, 384, 23])
    assert sample_transformed[SEG_KEY].shape == torch.Size([1, 384, 384, 23])


if __name__ == "__main__":
    test_get_base_transforms()
