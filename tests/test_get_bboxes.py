import torch

from pipeline.utils import get_bboxes


def test_get_bboxes_does_not_mutate_input():
    yx_locations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    original = yx_locations.clone()
    height = torch.tensor([2.0, 4.0])
    width = torch.tensor([2.0, 4.0])
    offset_x = torch.tensor([0.0, 0.0])
    offset_y = torch.tensor([0.0, 0.0])
    stride = 4
    _ = get_bboxes(yx_locations, height, width, offset_x, offset_y, stride=stride)
    assert torch.equal(yx_locations, original), "yx_locations was mutated in place"
