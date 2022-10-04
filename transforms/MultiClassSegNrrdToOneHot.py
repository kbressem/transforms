import re
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform, Transform
from monai.transforms.utils import TransformBackends, ensure_tuple_rep
from monai.utils import convert_data_type

import logging

logger = logging.getLogger(__name__)


class MulticlassSegNrrdToOneHot(Transform):
    """
    Convert seg.nrrd file to one hot format and adjust the size to match the size of the reference image.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self) -> None:
        pass

    def __call__(
        self, image: NdarrayOrTensor, label: NdarrayOrTensor, label_meta_dict: dict
    ) -> Tuple[NdarrayOrTensor, Dict]:
        """
        Args:
            image: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            label: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            label_meta_dict: former seg.nrrd header as dict

        Returns:
            A tuple of the label volume in one-hot format and the updated meta dict.

        """
        label, dtype, _ = convert_data_type(label, np.ndarray)
        segment_names = self._segment_names_from_label_meta_dict(label_meta_dict)
        n_seg = len(segment_names)
        one_hot = np.zeros((n_seg, *image.shape[1:]))
        for i, name in enumerate(segment_names):
            (
                new_layer_id,
                layer,
                label_value,
                extent,
                offset,
            ) = self._extract_info_by_name(label_meta_dict, name)
            if -1 in extent:
                continue  # skip empty segments
            segment = label[layer] == label_value
            segment_indices = [
                slice(e1, e2 + 1) for e1, e2 in zip(extent[0::2], extent[1::2])
            ]
            onehot_indices = [i] + [
                slice(o + e1, o + e2 + 1)
                for o, e1, e2 in zip(offset, extent[0::2], extent[1::2])
            ]
            one_hot[tuple(onehot_indices)] = segment[tuple(segment_indices)]
            label_meta_dict = self._update_meta_dict(label_meta_dict, i)

        label_meta_dict["spatial_shape"] = one_hot.shape[1:]
        one_hot, *_ = convert_data_type(one_hot, dtype)
        return (one_hot, label_meta_dict)

    def _update_meta_dict(self, label_meta_dict: dict, seg_id: int) -> dict:
        """Update Segment LabelValue and Segment Layer according to new values
        in the one hot encoded array.

        """
        # TODO: Update affine
        label_meta_dict = (label_meta_dict.copy())  # avoid weird bug where dict is changed inplace
        label_meta_dict[f"Segment{seg_id}_LabelValue"] = "1"
        label_meta_dict[f"Segment{seg_id}_Layer"] = str(seg_id)
        return label_meta_dict

    def _segment_names_from_label_meta_dict(self, label_meta_dict: dict) -> List[str]:
        """Read the names of segments from the label meta dict

        Args:
            label_meta_dict: Former NRRD header as dict

        Returns:
            List of segment names

        """
        segments = []
        for k in label_meta_dict.keys():
            m = re.match("Segment[0-9]+_Name$", k)
            if m:
                segments.append(label_meta_dict[k])
        return segments

    def _extract_info_by_name(
        self, label_meta_dict: dict, name: str
    ) -> Tuple[int, int, int, Tuple[Tuple[int, int], ...], Tuple[int, ...]]:
        """From the NRRD header, extract the layer number (int),
        label value (int), extend and offsett of the segment by segment name.

        Segment ID (int):
            ID of the segment in the header
        Layer number (int):
            Number of the layer
        Label value (int):
            Value that maps to the label in the layer
        Extent (list):
            The extent of the segmentation in width (x), height (y), depth (z)
            [x_start, x_end, y_start, y_end, y_start, y_end]
        Offset (list):
            Additonal Offset from the left upper image region
            [x, y, z]

        Args:
            label_meta_dict: former seg.nrrd header as dict
            name: name of the segment as str

        Returns
            A tuple of: segment ID, layer number, label value, extent, offset

        """
        segment_names = self._segment_names_from_label_meta_dict(label_meta_dict)
        try:
            seg_id = segment_names.index(name)
        except ValueError:
            raise ValueError(f"{name} not among segment names")

        layer = int(label_meta_dict[f"Segment{seg_id}_Layer"])
        label_value = int(label_meta_dict[f"Segment{seg_id}_LabelValue"])
        extent = label_meta_dict[f"Segment{seg_id}_Extent"]
        extent = list(map(int, extent.split(" ")))
        offset = label_meta_dict["Segmentation_ReferenceImageExtentOffset"].split(" ")
        offset = list(map(int, offset))
        return seg_id, layer, label_value, extent, offset


class MulticlassSegNrrdToOneHotd(MapTransform):
    """Dictionary-based wrapper of :py:class:`MulticlassSegNrrdToOneHot`"""

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
        self.adjuster = MulticlassSegNrrdToOneHot()
        self.ref_image_key = ref_image_key
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, meta_key, meta_key_postfix in zip(
            self.keys, self.meta_keys, self.meta_key_postfix
        ):
            values, meta = self.adjuster(
                d[self.ref_image_key],
                d[key],
                d[meta_key or f"{key}_{meta_key_postfix}"],
            )
            d[key] = values
            d[meta_key or f"{key}_{meta_key_postfix}"] = meta
        return d