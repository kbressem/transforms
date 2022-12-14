from typing import Dict, Hashable, Mapping, Optional, Union

import torch
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.data.meta_tensor import MetaTensor
from monai.networks.layers import apply_filter
from monai.transforms import MapTransform, RandomizableTransform, Transform
from monai.utils import convert_data_type, convert_to_tensor
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

    Laplacian filtering for outline detection in images. Can be used to transform labels to contours.
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

    ## Sobel kernel
    > `kernel={'sobel_h', 'sobel_w', ''sobel_d}`

    Edge detection with sobel kernel, along the h,w or d axis of tensor.
    Example 2D kernel (5x5) for `sobel_w`:

            [-0.25, -0.20,  0.00,  0.20,  0.25]
            [-0.40, -0.50,  0.00,  0.50,  0.40]
            [-0.50, -1.00,  0.00,  1.00,  0.50]
            [-0.40, -0.50,  0.00,  0.50,  0.40]
            [-0.25, -0.20,  0.00,  0.20,  0.25]

    ## Sharpen kernel
    > `kernel="sharpen"`

    Sharpen an image with a 2D or 3D kernel.
    Example 2D kernel (5x5):

            [ 0.,  0., -1.,  0.,  0.]
            [-1., -1., -1., -1., -1.]
            [-1., -1., 17., -1., -1.]
            [-1., -1., -1., -1., -1.]
            [ 0.,  0., -1.,  0.,  0.]
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    supported_kernels = sorted(["mean", "laplacian", "elliptical", "sobel_w", "sobel_h", "sobel_d", "sharpen"])

    def __init__(self, kernel: Union[str, NdarrayOrTensor], kernel_size: Optional[int] = None) -> None:

        if isinstance(kernel, str):
            assert kernel_size, "`kernel_size` must be specified when specifying kernels by string."
            assert kernel_size % 2 == 1, "`kernel_size` should be a single uneven integer."
            if kernel not in self.supported_kernels:
                raise NotImplementedError(f"{kernel}. Supported kernels are {self.supported_kernels}.")
        else:
            assert kernel.ndim in [1, 2, 3], "Only 1D, 2D, and 3D kernels are supported"
            kernel = convert_to_tensor(kernel, dtype=torch.float32)
            self._assert_all_values_uneven(kernel.shape)

        self.kernel = kernel
        self.kernel_size = kernel_size

    def __call__(self, img: NdarrayOrTensor, meta_dict: Optional[Mapping] = None) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to apply filter to with shape: [channels, height, width[, depth]]
            meta_dict: An optional dictionary with metadata

        Returns:
            A MetaTensor with the same shape as `img` and identical metadata
        """
        if isinstance(img, MetaTensor):
            meta_dict = img.meta
        img_, prev_type, device = convert_data_type(img, torch.Tensor)
        ndim = img_.ndim - 1  # assumes channel first format
        if isinstance(self.kernel, str):
            self.kernel = self._create_kernel_from_string(self.kernel, self.kernel_size, ndim)
        img_ = img_.unsqueeze(0)
        img_ = apply_filter(img_, self.kernel)  # batch, channels, H[, W, D] is required for img_
        img_ = img_[0]
        if meta_dict:
            img_ = MetaTensor(img_, meta_dict)
        else:
            img_, *_ = convert_data_type(img_, prev_type, device)
        return img_

    def _assert_all_values_uneven(self, x: tuple) -> None:
        for value in x:
            assert value % 2 == 1, f"Only uneven kernels are supported, but kernel size is {x}"

    def _create_kernel_from_string(self, name, size, ndim) -> torch.Tensor:
        """Create an `ndim` kernel of size `(size, ) * ndim`."""
        func = getattr(self, f"_create_{name}_kernel")
        kernel = func(size, ndim)
        return kernel.to(torch.float32)

    def _create_mean_kernel(self, size, ndim) -> torch.Tensor:
        """Create a torch.Tensor with shape (size, ) * ndim with all values equal to `1`"""
        return torch.ones([size] * ndim)

    def _create_laplacian_kernel(self, size, ndim) -> torch.Tensor:
        """Create a torch.Tensor with shape (size, ) * ndim.
        All values are `-1` except the center value which is size**ndim - 1
        """
        kernel = torch.zeros([size] * ndim).float() - 1  # make all -1
        center_point = tuple([size // 2] * ndim)
        kernel[center_point] = (size**ndim) - 1
        return kernel

    def _create_elliptical_kernel(self, size: int, ndim: int) -> torch.Tensor:
        """Create a torch.Tensor with shape (size, ) * ndim containing a circle/sphere of `1`"""
        radius = size // 2
        grid = torch.meshgrid(*[torch.arange(0, size) for _ in range(ndim)])
        squared_distances = torch.stack([(axis - radius) ** 2 for axis in grid], 0).sum(0)
        kernel = squared_distances <= radius**2
        return kernel

    def _sobel_2d(self, size) -> torch.Tensor:
        """Create a generic 2d sobel kernel"""
        numerator = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32).unsqueeze(0)
        denominator = numerator * numerator
        denominator = denominator + denominator.T
        denominator[:, size // 2] = 1.0  # to avoid division by zero
        return numerator / denominator

    def _sobel_3d(self, size) -> torch.Tensor:
        """Create a generic 3d sobel kernel"""
        kernel_2d = self._sobel_2d(size)
        kernel_3d = torch.stack((kernel_2d,) * size, -1)
        adapter = (size // 2) - torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32).abs()
        adapter = adapter / adapter.max() + 1  # scale between 1 - 2
        return kernel_3d * adapter

    def _create_sobel_w_kernel(self, size, ndim) -> torch.Tensor:
        """Sobel kernel in x/w direction for Tensor in shape (B,C)[WH[D]]"""
        if ndim == 2:
            kernel = self._sobel_2d(size)
        elif ndim == 3:
            kernel = self._sobel_3d(size)
        else:
            raise ValueError(f"Only 2 or 3 dimensional kernels are supported. Got {ndim}")
        return kernel

    def _create_sobel_h_kernel(self, size, ndim) -> torch.Tensor:
        """Sobel kernel in y/h direction for Tensor in shape (B,C)[WH[D]]"""
        kernel = self._create_sobel_w_kernel(size, ndim).transpose(0, 1)
        return kernel

    def _create_sobel_d_kernel(self, size, ndim) -> torch.Tensor:
        """Sobel kernel in z/d direction for Tensor in shape (B,C)[WHD]]"""
        assert ndim == 3, "Only 3 dimensional kernels are supported for `sobel_d`"
        return self._sobel_3d(size).transpose(1, 2)

    def _create_sharpen_kernel(self, size, ndim) -> torch.Tensor:
        """Create a torch.Tensor with shape (size, ) * ndim.
        The kernel contains a circle/sphere of `-1`, with the center value beeing
        the absolut sum of all non-zero elements in the kernel
        """
        kernel = self._create_elliptical_kernel(size, ndim)
        center_point = tuple([size // 2] * ndim)
        center_value = kernel.sum()
        kernel = kernel * -1
        kernel[center_point] = center_value
        return kernel


class FilterKernelTransformd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.FilterKernelTransform`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        kernel:
            A string specifying the kernel or a custom kernel as `torch.Tenor` or `np.ndarray`.
            Available options are: `mean`, `laplacian`, `elliptical`, `sobel_{w,h,d}``
        kernel_size:
            A single integer value specifying the size of the quadratic or cubic kernel.
            Computational complexity increases exponentially with kernel_size, which
            should be considered when choosing the kernel size.
        allow_missing_keys:
            Don't raise exception if key is missing.
    """

    backend = FilterKernelTransform.backend

    def __init__(
        self,
        keys: KeysCollection,
        kernel: Union[str, NdarrayOrTensor],
        kernel_size: Optional[int] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.filter = FilterKernelTransform(kernel, kernel_size)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.filter(d[key])
        return d


class RandFilterKernelTransform(RandomizableTransform):
    """Randomly apply a Filterkernel to the input data.
    Args:
        kernel:
            A string specifying the kernel or a custom kernel as `torch.Tenor` or `np.ndarray`.
            Available options are: `mean`, `laplacian`, `elliptical`, `gaussian``
            See below for short explanations on every kernel.
        kernel_size:
            A single integer value specifying the size of the quadratic or cubic kernel.
            Computational complexity scales to the power of 2 (2D kernel) or 3 (3D kernel), which
            should be considered when choosing kernel size.
        prob:
            Probability the transform is applied to the data
    """

    backend = FilterKernelTransform.backend

    def __init__(
        self, kernel: Union[str, NdarrayOrTensor], kernel_size: Optional[int] = None, prob: float = 0.1
    ) -> None:
        super().__init__(prob)
        self.filter = FilterKernelTransform(kernel, kernel_size)

    def __call__(self, img: NdarrayOrTensor, meta_dict: Optional[Mapping] = None) -> NdarrayOrTensor:
        """
        Args:
            img: torch tensor data to apply filter to with shape: [channels, height, width[, depth]]
            meta_dict: An optional dictionary with metadata

        Returns:
            A MetaTensor with the same shape as `img` and identical metadata
        """
        self.randomize(None)
        if self._do_transform:
            img = self.filter(img)
        return img


class RandFilterKernelTransformd(MapTransform, RandomizableTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandomFilterKernel`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        kernel:
            A string specifying the kernel or a custom kernel as `torch.Tenor` or `np.ndarray`.
            Available options are: `mean`, `laplacian`, `elliptical`, `sobel_{w,h,d}``
        kernel_size:
            A single integer value specifying the size of the quadratic or cubic kernel.
            Computational complexity increases exponentially with kernel_size, which
            should be considered when choosing the kernel size.
        prob:
            Probability the transform is applied to the data
        allow_missing_keys:
            Don't raise exception if key is missing.
    """

    backend = FilterKernelTransform.backend

    def __init__(
        self,
        keys: KeysCollection,
        kernel: Union[str, NdarrayOrTensor],
        kernel_size: Optional[int] = None,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.filter = FilterKernelTransform(kernel, kernel_size)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.filter(d[key])
        return d
