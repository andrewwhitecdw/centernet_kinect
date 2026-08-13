import random
import torch
import pytest

from pipeline.utils import get_image


class FakeDataset:
    def __init__(self, data):
        self.dataset = data


class FakeModelSetup:
    def __init__(self):
        self.train_dataset = FakeDataset([(torch.rand(3, 4), torch.rand(5)) for _ in range(3)])
        self.valid_dataset = FakeDataset([(torch.rand(3, 4), torch.rand(5)) for _ in range(2)])


def test_get_image_index_within_bounds():
    random.seed(0)
    setup = FakeModelSetup()
    for _ in range(50):
        img, gt = get_image(setup, train=True)
        assert img.shape[0] == 1
        assert gt.shape[0] == 1
        img, gt = get_image(setup, train=False)
        assert img.shape[0] == 1
