from cv2 import transform
import torch
import unittest
from parameterized import parameterized
from transforms import FilterKernelTransform

expected_kernels = {

    "mean" : torch.tensor([[1,1,1], [1,1,1], [1,1,1]]), 
    "laplace" : torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]), 
    "elliptical" : torch.tensor([[0,1,0], [1,1,1], [0,1,0]]), 
    "sobel_h": torch.tensor([-1,0,1], [-2,0,2], [-1,0,1]), 
    "sobel_v": torch.tensor([-1,-2,-1], [0,0,0], [1,2,1]), 

}


class TestGaussianSmooth(unittest.TestCase):

    def text_init(self): 
        transform = FilterKernelTransform()