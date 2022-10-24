import logging
from typing import Dict, Hashable, Optional, Tuple

import numpy as np
import torch
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform, Transform
from monai.transforms.utils import ensure_tuple_rep
from monai.utils import convert_data_type, convert_to_tensor
from scipy import ndimage
from torch.nn import functional as F  # noqa F401

logger = logging.getLogger(__name__)


class PathologyLabelSoften(Transform):
    """Convert sharp one-hot encoded labels to soft labels
    Rationale:
        The rAIdiance pathology segmentation works with 3 levels of confidence:
            1/3 - low severity of pathology / not sure if pathology present
            2/3 - medium severity / pathology likely present
            3/3 - high severity / pathology likely present
        These labels can be converted to one-hot format using `MulticlassSegNrrdToOneHot`, however
        for model training this is not suitable. 1/3 are weak segmentations and more likely to be wrong
        than 3/3 segmentations. Therefore, if the models predictions do not match 1/3 segmentations, the
        loss should increase by the same amount as when the model does not match 3/3 segmentations.
        With `PathologyLabelSoften` even finer increments in the labels can be achieved, leading to a smoother
        loss landscape.
    """

    def __init__(self, increments=10) -> None:
        self.num_increment_steps = 20
        assert increments > 0 and increments < self.num_increment_steps, "Increments should be between 1 and 19"
        self.increments = increments
        self.classes = ("effusion", "infiltration", "atelectasis")
        self.severity = ("_1/3", "_2/3", "_3/3")
        self.soft_classes = [f"{c}{s}" for c in self.classes for s in self.severity]

    def __call__(self, label: NdarrayOrTensor, meta_dict: Dict) -> Tuple[NdarrayOrTensor, Dict]:
        label, prev_type, device = convert_data_type(label, np.ndarray)
        # labels are pseudo-3d, means format is C [[H, W], 1]
        # squeezing makes working with labels easier
        label = label.squeeze()
        label = self._sort_labels(label, meta_dict)
        label = self._create_soft_labels(label)
        label = np.expand_dims(label, -1)  # undo squeeze
        meta_dict = self._normalize_dict(label, meta_dict)
        label, *_ = convert_data_type(label, prev_type, device)
        return label, meta_dict

    def _sort_labels(self, label: np.ndarray, meta_dict: Dict) -> Tuple[np.ndarray, Dict]:
        """Create fixed order of labels for save indexing

        Has side effects: Deletes label array

        Args:
            label: an array coming from a seg.nrrd file
            meta_dict: A dict created from a seg.nrrd header.
                Relevant keys for this function are `SegmentX_LabelValue` and SegmentX_Layer` where
                `X` is an integer. These indicate the layer in the array, the label is found and the
                value it has.
        Returns:
            np.ndarray, the updated array with labels in fixed order.
        """
        n_labels = len(self.soft_classes) + 1
        new_label = np.zeros((n_labels, *label.shape[1:]))
        expected_segments = self.soft_classes + ["pneumothorax"]

        # check if expected_segments of expected segments is a subset of the meta keys
        for expected_segment in expected_segments:
            assert (
                expected_segment in meta_dict.values()
            ), f"The segmentation is missing key {expected_segment}. It only has: {meta_dict.values()}"

        for i, c in enumerate(expected_segments):
            idx = self._get_layer_id_by_value(meta_dict, c)
            try:
                label_value = int(meta_dict[f"Segment{idx}_LabelValue"])
                label_layer = int(meta_dict[f"Segment{idx}_Layer"])
                new_label[i] = (label[label_layer] == label_value).astype(np.uint8)
            except ValueError:
                new_label[i] = 0
                logger.warning(f"{c} not found in meta_dict keys, was it accidentially removed?")
        del label
        return new_label

    def _get_layer_id_by_value(self, meta_dict: Dict, value: str) -> int:
        """Get key from `meta_dict` corresponding to `value`, then get layer-id from key"""

        # Find idx
        idx = None
        for i, (meta_key, meta_val) in enumerate(meta_dict.items()):
            if meta_val == value and meta_key.endswith("_Name"):  # not the one ending with "_ID"
                idx = i
        assert idx is not None, f"Could not find {value} in meta_dict ID keys: {list(meta_dict.items())}"

        key = list(meta_dict.keys())[idx]
        return int(key.replace("Segment", "").replace("_Name", ""))

    def _create_soft_labels(self, labels: np.ndarray) -> np.ndarray:
        """Create fine increments at borders of pathology segmentations"""
        *multilabels, pneumothorax = self._split_by_label(labels)
        multilabels = [self._adapt_label_areas(label) for label in multilabels]
        for i, label in enumerate(multilabels):
            label = [self._soften_label(sub_label) for sub_label in label]
            multilabels[i] = self._smooth(np.mean(label, 0))
        multilabels = np.stack(multilabels, 0)
        labels = np.concatenate([multilabels, pneumothorax], 0)
        return labels

    def _smooth(self, label: np.ndarray) -> np.ndarray:
        """Apply median smooth to 2d numpy array"""
        label = convert_to_tensor(label)
        label = label.unsqueeze(0).unsqueeze(0).float()
        ks = 3
        median_kernel = torch.ones(1, 1, ks, ks) / ks**2
        label = F.conv2d(label, median_kernel, padding=ks // 2)
        return label.squeeze().numpy()

    def _soften_label(self, label: np.ndarray) -> np.ndarray:
        """Soften the border of a binary label by pyramide-like pooling of label areas"""
        label, n = ndimage.label(label)  # identify different independent areas
        soft_label = np.zeros(label.shape)
        for i in range(n):
            upper_left, bottom_right = self._extract_bbox(label == i + 1)

            # if x1==x2 or y1==y2, the area is only zero pixel wide, which is not supported
            if (upper_left == bottom_right).any():
                continue

            x, y = np.ogrid[
                upper_left[0] : bottom_right[0],  # noqa E203
                upper_left[1] : bottom_right[1],  # noqa E203
            ]

            # print("x, y", x.shape, y.shape)
            if x.shape == (1, 1) or y.shape == (1, 1):
                # print("skip")
                continue

            partial_label = [
                self._interpolate_and_pad(label[x, y], j / self.num_increment_steps)
                for j in range(self.num_increment_steps - self.increments, self.num_increment_steps)
            ]
            soft_label[x, y] = np.mean(np.stack(partial_label, 0), 0)
        return soft_label

    def _interpolate_and_pad(self, label: np.ndarray, relative_size_reduction: float) -> np.ndarray:
        """Shrink a binary area but keep the center of mass at the same position in the array

        label: shape: (139, 188)
        relative_size_reduction: 0.5

        Returns: shape: (139, 188)
        """
        com = ndimage.center_of_mass(label)
        com = [int(i) for i in com]

        relative_pad_x = com[0] / label.shape[0]
        relative_pad_y = com[1] / label.shape[1]

        label = convert_to_tensor(label)
        old_size = label.shape
        new_size = [int(sz * relative_size_reduction) for sz in old_size]
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(label, new_size, mode="bilinear")

        pad_x = old_size[0] - new_size[0]
        pad_y = old_size[1] - new_size[1]
        pad_x1 = int(pad_x * relative_pad_x)
        pad_x2 = int(pad_x * (1 - relative_pad_x))

        pad_y1 = int(pad_y * relative_pad_y)
        pad_y2 = int(pad_y * (1 - relative_pad_y))
        label = F.pad(label, (pad_x1, pad_x2, pad_y1, pad_y2))
        # padding does not create the exact same size due to rounding errors
        label = F.interpolate(label, old_size)
        return label.squeeze().numpy()

    def _adapt_label_areas(self, label: np.ndarray) -> np.ndarray:
        """Pathology labels are additive. If 3/3 is present, so are 1/3 and 2/3.
        However, this is not reliably done in the segmentations and thus needs to be added.

        Side effect: Changes input label
        """
        mask_2of3_is_on = label[1] == 1
        label[0][mask_2of3_is_on] = 1
        mask_3of3_is_on = label[2] == 1
        label[0][mask_3of3_is_on] = 1
        label[1][mask_3of3_is_on] = 1
        return label

    def _extract_bbox(self, label: np.ndarray) -> np.ndarray:
        """Get upper left and bottom right corner of area"""
        indices = np.argwhere(label)
        upper_left = np.min(indices, axis=0)
        bottom_right = np.max(indices, axis=0)
        return upper_left, bottom_right

    def _split_by_label(
        self,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separate pathology instances from label"""
        effusion = label[0:3]
        infiltration = label[3:6]
        atelectasis = label[6:9]
        pneumothorax = label[9:10]
        return effusion, infiltration, atelectasis, pneumothorax

    def _normalize_dict(self, label: np.ndarray, meta_dict: Dict) -> Dict:
        """Update SegmentX_ tags in the `meta_dict` with new names and extents"""
        for k in list(meta_dict.keys()):
            if "Segment" in k:
                meta_dict.pop(k)

        for i, name in enumerate((*self.classes, "pneumothorax")):
            meta_dict[f"Segment{i}_ID"] = name
            meta_dict[f"Segment{i}_Name"] = name
            meta_dict[f"Segment{i}_Layer"] = i
            try:
                upper_left, bottom_right = self._extract_bbox(label[i])
                extent = (
                    f"{upper_left[0]} {bottom_right[0]} {upper_left[1]} "
                    f"{bottom_right[1]} {upper_left[2]} {bottom_right[2]}"
                )
            except ValueError:
                extent = "0 0 0 0 0 0"
            meta_dict[f"Segment{i}_Extent"] = extent
        return meta_dict


class PathologyLabelSoftend(MapTransform):
    """Dictionary-based wrapper of :py:class:`PathologyLabelSoften`"""

    def __init__(
        self,
        keys: KeysCollection,
        ref_image_key: str,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            ref_image_key: key for the reference image to derive the size of the segmentation from
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None and `key_{postfix}` was used to store the metadata in `LoadImaged`.
                So need the key to extract metadata for channel dim information, default is `meta_dict`.
                For example, for data with key `image`, metadata by default is in `image_meta_dict`.
        """
        super().__init__(keys)
        self.adjuster = PathologyLabelSoften()
        self.ref_image_key = ref_image_key
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, meta_key, meta_key_postfix in zip(self.keys, self.meta_keys, self.meta_key_postfix):
            values, meta = self.adjuster(d[key], d[meta_key or f"{key}_{meta_key_postfix}"])
            d[key] = values
            d[meta_key or f"{key}_{meta_key_postfix}"] = meta
        return d
