import unittest

import torch
from parameterized import parameterized

from transforms.FilterKernelTransform import (
    FilterKernelTransform,
    FilterKernelTransformd,
    RandFilterKernelTransform,
    RandFilterKernelTransformd,
)

EXPECTED_KERNELS = {
    "mean": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).float(),
    "laplacian": torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float(),
    "elliptical": torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).float(),
    "sobel_w": torch.tensor([[-0.5, 0, 0.5], [-1, 0, 1], [-0.5, 0, 0.5]]).float(),
    "sobel_h": torch.tensor([[-0.5, -1, -0.5], [0, 0, 0], [0.5, 1, 0.5]]).float(),
    "sharpen": torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).float(),
}

SUPPORTED_KERNELS = ["mean", "laplacian", "elliptical", "sobel_w", "sobel_h", "sobel_d", "sharpen"]
SAMPLE_IMAGE_2D = torch.randn(1, 10, 10)
SAMPLE_IMAGE_3D = torch.randn(1, 10, 10, 10)
SAMPLE_DICT = {"image_2d": SAMPLE_IMAGE_2D, "image_3d": SAMPLE_IMAGE_3D}


class TestFilterKernelTransform(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = FilterKernelTransform(kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = FilterKernelTransform(kernel_name)

    def test_init_from_array(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = FilterKernelTransform(torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = FilterKernelTransform(torch.ones(3, 3, 3, 3))

    @parameterized.expand(EXPECTED_KERNELS.keys())
    def test_2d_kernel_correctness(self, kernel_name):
        "Test correctness of kernels (2d only)"
        tfm = FilterKernelTransform(kernel_name, kernel_size=3)
        kernel = tfm._create_kernel_from_string(kernel_name, size=3, ndim=2).squeeze()
        torch.testing.assert_allclose(kernel, EXPECTED_KERNELS[kernel_name])

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = FilterKernelTransform(kernel_name, 3)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_IMAGE_2D)
            self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = FilterKernelTransform(kernel_name, 3)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_3D.shape)


class TestFilterKernelTransformDict(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string_dict(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = FilterKernelTransformd("image", kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = FilterKernelTransformd(self.image_key, kernel_name)

    def test_init_from_array_dict(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = FilterKernelTransformd("image", torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = FilterKernelTransformd(self.image_key, torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = FilterKernelTransformd("image_2d", kernel_name, 3)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_DICT)
            self.assertEqual(out_tensor["image_2d"].shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = FilterKernelTransformd("image_3d", kernel_name, 3)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_3d"].shape, SAMPLE_IMAGE_3D.shape)


class TestRandFilterKernelTransform(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = RandFilterKernelTransform(kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandFilterKernelTransform(kernel_name)

    def test_init_from_array(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = RandFilterKernelTransform(torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandFilterKernelTransform(torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_1(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = RandFilterKernelTransform(kernel_name, 3, 1)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_IMAGE_2D)
            self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_1(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = RandFilterKernelTransform(kernel_name, 3, 1)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_3D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_0(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = RandFilterKernelTransform(kernel_name, 3, 0)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_IMAGE_2D)
            torch.testing.assert_allclose(out_tensor, SAMPLE_IMAGE_2D)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_0(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = RandFilterKernelTransform(kernel_name, 3, 0)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        torch.testing.assert_allclose(out_tensor, SAMPLE_IMAGE_3D)


class TestRandFilterKernelTransformDict(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string_dict(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = RandFilterKernelTransformd("image", kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandFilterKernelTransformd("image", kernel_name)

    def test_init_from_array_dict(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = RandFilterKernelTransformd("image", torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandFilterKernelTransformd("image", torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_1(self, kernel_name):
        filter = RandFilterKernelTransformd("image_2d", kernel_name, 3, 1.0)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_DICT)
            self.assertEqual(out_tensor["image_2d"].shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_1(self, kernel_name):
        filter = RandFilterKernelTransformd("image_3d", kernel_name, 3, 1.0)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_3d"].shape, SAMPLE_IMAGE_3D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_0(self, kernel_name):
        filter = RandFilterKernelTransformd("image_2d", kernel_name, 3, 0.0)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_DICT)
            torch.testing.assert_allclose(out_tensor["image_2d"].shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_0(self, kernel_name):
        filter = RandFilterKernelTransformd("image_3d", kernel_name, 3, 0.0)
        out_tensor = filter(SAMPLE_DICT)
        torch.testing.assert_allclose(out_tensor["image_3d"].shape, SAMPLE_IMAGE_3D.shape)
