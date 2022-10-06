from typing import Mapping, Optional, Union

import torch
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.networks.layers import apply_filter
from monai.transforms import Transform
from monai.utils import convert_to_tensor
from monai.utils.enums import TransformBackends


class FilterKernelTransform(Transform):
    """
    Applies a kernel transformation to the input image.

    Args:
        kernel:
            A string specifying the kernel or a custom kernel as `torch.Tenor` or `np.ndarray`.
            Available options are: `mean`, `laplacian`, `elliptical`, `gaussian``
            See below for short explanations on every kernel.
        kernel_size:
            A single integer value specifying the size of the quadratic or cubic kernel.
            Computational complexity scales to the power of 2 (2D kernel) or 3 (3D kernel), which
            should be considered when choosing kernel size.
        convert_one_hot:
            Convert image to one_hot format

    Raises:
        AssertionError: When `kernel` is a string  and `kernel_size` is not specified
        AssertionError: When `kernel_size` is not an uneven integer
        AssertionError: When `kernel` is an array and `ndim` is not in [1,2,3]
        AssertionError: When `kernel` is an array and any dimension has an even shape
        NotImplementedError: When `kernel` is a string and not in `self.supported_kernels`


    ## Mean kernel
    > `kernel='mean'`

    Mean filtering can smooth edges and remove aliasing artifacts in an segmentation image.
    Example 2D kernel (5 x 5):

            [1, 1, 1, 1, 1]
            [1, 1, 1, 1, 1]
            [1, 1, 1, 1, 1]
            [1, 1, 1, 1, 1]
            [1, 1, 1, 1, 1]

    If smoothing labels with this kernel, ensure they are in one-hot format.

    ## Laplacian kernel
    > `kernel='laplacian'`

    Laplacian filtering for edge detection in images. Can be used to transform labels to contours.
    Example 2D kernel (5x5):

            [-1., -1., -1., -1., -1.]
            [-1., -1., -1., -1., -1.]
            [-1., -1., 24., -1., -1.]
            [-1., -1., -1., -1., -1.]
            [-1., -1., -1., -1., -1.]

    ## Elliptical kernel
    > `kernel='elliptical'`

    An elliptical kernel can be used to dilate labels or label-contours.
    Example 2D kernel (5x5):

            [0., 0., 1., 0., 0.]
            [1., 1., 1., 1., 1.]
            [1., 1., 1., 1., 1.]
            [1., 1., 1., 1., 1.]
            [0., 0., 1., 0., 0.]
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    supported_kernels = sorted(["mean", "laplacian", "elliptical"])

    def __init__(
        self, kernel: Union[str, NdarrayOrTensor], kernel_size: Optional[int] = None, convert_one_hot: bool = False,
    ) -> None:

        if isinstance(kernel, str):
            assert kernel_size, "`kernel_size` must be specified when specifying kernels by string."
            assert kernel_size % 2 == 1, "`kernel_size` should be a single uneven integer."
            if kernel not in self.supported_kernels:
                raise NotImplementedError(f"{kernel}. Supported kernels are {self.supported_kernels}.")
        else:
            assert kernel.ndim in [1, 2, 3], "Only 1D, 2D, and 3D kernels are supported"
            self._assert_all_values_even(kernel)
            kernel = convert_to_tensor(kernel, dtype=torch.float32)

        self.kernel = kernel
        self.kernel_size = kernel_size
        self.convert_one_hot = convert_one_hot

    def __call__(self, img: NdarrayOrTensor, meta_dict: Optional[Mapping] = None) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to apply filter to with shape: [channels, height, width[, depth]]

        Returns:
            A MetaTensor with the same shape as `img` and identical metadata
        """
        if isinstance(img, MetaTensor):
            meta_dict = img.meta
        img_ = convert_to_tensor(img, track_meta=False)
        ndim = img_.shape - 1  # assumes channel first format
        if isinstance(self.kernel, str):
            self.kernel = self._create_kernel_from_string(self.kernel, self.kernel_size, ndim)
        img_ = apply_filter(img_, self.kernel)
        if meta_dict:
            img_ = MetaTensor(img_, meta_dict)
        return img_

    def _assert_all_values_even(self, x: tuple) -> None:
        for value in x:
            assert value % 2 == 1, f"Only uneven kernels are supported, but kernel size is {x}"

    def _create_kernel_from_string(self, name, size, ndim):
        "Create an `ndim` kernel of size `(size, ) * ndim`."
        func = getattr(self, f"_create_{name}_kernel")
        kernel = func(size, ndim)
        return kernel.to(torch.float32)

    def _create_mean_kernel(self, size, ndim):
        return torch.ones([1, 1] + [size] * ndim)

    def _create_laplacian_kernel(self, size, ndim):
        kernel = torch.ones([1, 1] + [size] * ndim).float() - 2  # make all -1
        center_point = tuple([0, 0] + [size // 2] * ndim)
        kernel[center_point] = (size ** ndim) - 1
        return kernel

    def _create_elliptical_kernel(size: int, ndim: int) -> torch.Tensor:
        radius = size // 2
        grid = torch.meshgrid(*[torch.arange(0, size) for _ in range(ndim)])
        squared_distances = torch.stack([(axis - radius) ** 2 for axis in grid], 0).sum(0)
        kernel = squared_distances <= radius ** 2
        return kernel

    def _create_sobel_kernel(self, size, ndim) -> torch.Tensor:
        # kernel function adapted from `monai.transforms.SobelGradients` by @drbeh

        numerator = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
        denominator = numerator * numerator
        denominator = denominator + denominator.T
        denominator[:, size // 2] = 1.0  # to avoid division by zero
        kernel = numerator / denominator
        return kernel
