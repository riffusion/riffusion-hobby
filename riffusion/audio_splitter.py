import typing as T

import numpy as np
import pydub
import torch
import torchaudio
from torchaudio.transforms import Fade

from riffusion.util import audio_util


class AudioSplitter:
    """
    Split audio into instrument stems like {drums, bass, vocals, etc.}

    See:
        https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html
    """

    def __init__(
        self,
        segment_length_s: float = 10.0,
        overlap_s: float = 0.1,
        device: str = "cuda",
    ):
        self.segment_length_s = segment_length_s
        self.overlap_s = overlap_s
        self.device = device

        self.model = self.load_model().to(device)

    @staticmethod
    def load_model(model_path: str = "models/hdemucs_high_trained.pt") -> torchaudio.models.HDemucs:
        """
        Load the trained HDEMUCS pytorch model.
        """
        # NOTE(hayk): The sources are baked into the pretrained model and can't be changed
        model = torchaudio.models.hdemucs_high(sources=["drums", "bass", "other", "vocals"])

        path = torchaudio.utils.download_asset(model_path)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def split(self, audio: pydub.AudioSegment) -> T.Dict[str, pydub.AudioSegment]:
        """
        Split the given audio segment into instrument stems.
        """
        if audio.channels == 1:
            audio_stereo = audio.set_channels(2)
        elif audio.channels == 2:
            audio_stereo = audio
        else:
            raise ValueError(f"Audio must be stereo, but got {audio.channels} channels")

        # Get as (samples, channels) float numpy array
        waveform_np = np.array(audio_stereo.get_array_of_samples())
        waveform_np = waveform_np.reshape(-1, audio_stereo.channels)
        waveform_np_float = waveform_np.astype(np.float32)

        # To torch and channels-first
        waveform = torch.from_numpy(waveform_np_float).to(self.device)
        waveform = waveform.transpose(1, 0)

        # Normalize
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()

        # Split
        sources = self.separate_sources(
            waveform[None],
            sample_rate=audio.frame_rate,
        )[0]

        # De-normalize
        sources = sources * ref.std() + ref.mean()

        # To numpy
        sources_np = sources.cpu().numpy().astype(waveform_np.dtype)

        # Convert to pydub
        stem_segments = [
            audio_util.audio_from_waveform(waveform, audio.frame_rate) for waveform in sources_np
        ]

        # Convert back to mono if necessary
        if audio.channels == 1:
            stem_segments = [stem.set_channels(1) for stem in stem_segments]

        return dict(zip(self.model.sources, stem_segments))

    def separate_sources(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 44100,
    ):
        """
        Apply model to a given waveform in chunks. Use fade and overlap to smooth the edges.
        """
        batch, channels, length = waveform.shape

        chunk_len = int(sample_rate * self.segment_length_s * (1 + self.overlap_s))
        start = 0
        end = chunk_len
        overlap_frames = self.overlap_s * sample_rate
        fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

        final = torch.zeros(batch, len(self.model.sources), channels, length, device=self.device)

        # TODO(hayk): Improve this code, which came from the torchaudio docs
        while start < length - overlap_frames:
            chunk = waveform[:, :, start:end]
            with torch.no_grad():
                out = self.model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0

        return final
