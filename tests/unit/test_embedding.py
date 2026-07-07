"""Unit tests for CLIP embedding helpers (Transformers v4/v5 compatibility)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services.embedding import clip_feature_tensor, image_embedding


@dataclass
class FakeModelOutput:
    pooler_output: torch.Tensor


def test_clip_feature_tensor_accepts_raw_tensor():
    tensor = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(clip_feature_tensor(tensor), tensor)


def test_clip_feature_tensor_extracts_pooler_output():
    tensor = torch.tensor([[0.5, 0.5, 0.5]])
    output = FakeModelOutput(pooler_output=tensor)
    assert torch.equal(clip_feature_tensor(output), tensor)


def test_clip_feature_tensor_extracts_tuple_first_element():
    tensor = torch.tensor([[1.0, 0.0]])
    assert torch.equal(clip_feature_tensor((tensor,)), tensor)


def test_clip_feature_tensor_raises_on_unexpected_type():
    with pytest.raises(TypeError, match="Unexpected CLIP feature output"):
        clip_feature_tensor({"bad": "shape"})


@patch("app.services.embedding.preprocess_image")
@patch("app.services.embedding.load_clip_model")
def test_image_embedding_normalizes_pooler_output(mock_load, mock_preprocess):
    mock_preprocess.return_value = MagicMock()
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    model = MagicMock()
    model.get_image_features.return_value = FakeModelOutput(
        pooler_output=torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    )
    mock_load.return_value = (model, processor)

    with patch("app.services.embedding.get_device", return_value=torch.device("cpu")):
        embedding = image_embedding("/tmp/fake.png")

    assert len(embedding) == 2
    assert abs(embedding[0] - 0.6) < 1e-5
    assert abs(embedding[1] - 0.8) < 1e-5
