import random
import torch
import torchaudio
import librosa
import numpy as np
from typing import List
from torch import distributions


def background_aug(audio: torch.Tensor,
                   audio_back: List,
                   level: int = 20) -> torch.Tensor:
    noise = random.choice(audio_back)

    noise_level = torch.Tensor([level])
    noise_energy = torch.norm(noise)
    audio_energy = torch.norm(audio)

    alpha = (audio_energy / noise_energy) * torch.pow(10, -noise_level / 20)

    if noise.shape[1] > audio.shape[1]:
        pos = random.randint(0, noise.shape[1] - audio.shape[1])
        if audio.shape[1] + pos < noise.shape[1]:
            noise = noise[..., pos:audio.shape[1] + pos]
        else:
            noise = noise[..., :audio.shape[1]]
    else:
        ratio = np.ceil(audio.shape[1] / noise.shape[1]).astype(int)
        noise = noise.repeat((1, ratio))[..., :audio.shape[1]]

    return torch.clamp(audio + alpha * noise, -1, 1)


def volume_aug(wav: torch.Tensor,
               gain: float = .2,
               gain_type: str = 'amplitude') -> torch.Tensor:
    return torchaudio.transforms.Vol(gain=gain, gain_type=gain_type)(wav)


def time_stretch(wav: torch.Tensor,
                 stretch_ratio: float = 1.2) -> torch.Tensor:
    audio = librosa.effects.time_stretch(wav.numpy().squeeze(), rate=stretch_ratio)
    return torch.from_numpy(audio).unsqueeze(0)


def pitch_shift(wav: torch.Tensor,
                rate: int = 16000,
                ratio: float = 1.1) -> torch.Tensor:
    audio = librosa.effects.pitch_shift(wav.numpy().squeeze(),
                                        sr=rate, n_steps=ratio)
    return torch.from_numpy(audio).unsqueeze(0)


def gaussian_noise(wav: torch.Tensor,
                   noiser: torch.distributions = distributions.Normal(0, 0.01)) -> torch.Tensor:
    return wav + noiser.sample(wav.size())