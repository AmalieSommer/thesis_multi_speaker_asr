from multi_speaker_asr.models.diarization import SpeakerDiarizationPipeline
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pytest
import torch


# ------ FIXTURES -------
@pytest.fixture(scope='function')
def diarization(request):
    with patch("multi_speaker_asr.models.diarization.Pipeline.from_pretrained") as mock_from_pretrained:
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        mock_pipeline.to.return_value = mock_pipeline
        return SpeakerDiarizationPipeline(config=request.param), mock_pipeline

