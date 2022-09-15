import os
import math
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.utils.encoders import TextEncDec


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 file: str,
                 mode: str = 'train',
                 transform: Optional[List] = None,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 64,
                 time_mask: int = 35,
                 freq_mask: int = 15
                 ):
        self.data = pd.read_csv(file)
        self.transform = transform
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.mode = mode

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.mel_spec_cutout = nn.Sequential(
            self.melspec,
            torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask),
            torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask)
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        data, rate = torchaudio.load(f"../{self.data.loc[idx, 'path']}")
        data = data.squeeze()
        text = self.data.loc[idx, 'Expected']

        length = data.shape[0]
        pos = 0

        if self.transform and self.mode == 'train':
            pos = torch.randint(low=0, high=len(self.transform), size=(1,)).item()

            data = data.unsqueeze(0)
            data = self.transform[pos](data)
            length = data.shape[1]

            spec_len = math.ceil(int((length // 256) + 1) / 2)
            if spec_len / self.time_mask > 10:
                data = torch.log(self.melspec_cutout(data) + 1e-9)
            else:
                current_time_mask = 0.1 * spec_len
                self.mel_spec_cutout_time = nn.Sequential(
                    self.melspec,
                    torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask),
                    torchaudio.transforms.TimeMasking(time_mask_param=current_time_mask)
                )
                data = torch.log(self.melspec_cutout_time(data) + 1e-9)

            data = data.squeeze()

        else:
            data = torch.log(self.melspec(data) + 1e-9)

        return {'input': data.T, 'output': text, 'indices': idx, 'len': length, 'augs': pos}


class ASRDataModule(LightningDataModule):
    def __init__(
            self,
            train_data_path: str = "train_sorted.csv",
            test_data_path: str = "test_sorted.csv",
            val_data_path: str = "val_sorted.csv",
            train_test_split: Tuple[float, float, float] = (0.85, 0.15, 0.05),
            batch_size: int = 32,
            sample_size: int = 64,
            n_mels: int = 64,
            sample_rate: int = 16000,
            n_fft: int = 1024,
            n_hop: int = 256,
            freq_mask_param: int = 15,
            time_mask_param: int = 30,
            num_workers: int = 0,
            pin_memory: bool = False,
            data_shuffle: bool = True,
            augmentations: List = None,
            encoder: Any = TextEncDec()
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.encoder = self.hparams.encoder

    @property
    def num_classes(self) -> int:
        return len(self.encoder)

    def prepare_data(self):
        pass

        # download

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Dataset(
                file=self.hparams.train_data_path,
                transform=self.hparams.augmentations,
                sample_rate=self.hparams.sample_rate,
                n_fft=self.hparams.n_fft,
                hop_length=self.hparams.n_hop,
                n_mels=self.hparams.n_mels,
                time_mask=self.hparams.time_mask_param,
                freq_mask=self.hparams.freq_mask_param
            )

            self.data_val = Dataset(
                file=self.hparams.val_data_path,
                mode="test",
                transform=None,
                sample_rate=self.hparams.sample_rate,
                n_fft=self.hparams.n_fft,
                hop_length=self.hparams.n_hop,
                n_mels=self.hparams.n_mels,
                time_mask=self.hparams.time_mask_param,
                freq_mask=self.hparams.freq_mask_param
            )

            self.data_test = Dataset(
                file=self.hparams.test_data_path,
                mode="test",
                transform=None,
                sample_rate=self.hparams.sample_rate,
                n_fft=self.hparams.n_fft,
                hop_length=self.hparams.n_hop,
                n_mels=self.hparams.n_mels,
                time_mask=self.hparams.time_mask_param,
                freq_mask=self.hparams.freq_mask_param
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.sample_size,
            shuffle=self.hparams.data_shuffle,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.sample_size,
            shuffle=self.hparams.data_shuffle,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.sample_size,
            shuffle=self.hparams.data_shuffle,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def collate_fn(self, data) -> Tuple:
        wavs = []
        input_lens = []
        labels = []
        labels_lens = []
        indices = []
        augs = []

        data = sorted(data, key=lambda d: d['len'])

        for elem in data:
            wavs.append(elem['input'])
            input_lens.append(math.ceil(int((elem['len'] // 256) + 1) / 2))
            label = torch.Tensor(self.encoder.encode(elem['output']))
            labels.append(label)
            indices.append(elem['indices'])
            augs.append(elem['augs'])
            labels_lens.append(len(label))

        steps = self.hparams.sample_size // self.hparams.batch_size
        step_wavs = []
        step_input_lens = []
        step_labels = []
        step_labels_lens = []
        step_indices = []
        step_augs = []

        for i in range(steps):
            step_wavs.append(pad_sequence(
                wavs[self.hparams.batch_size * i: self.hparams.batch_size * (i + 1)],
                batch_first=True).permute(0, 2, 1)
                             )
            step_labels.append(pad_sequence(
                labels[self.hparams.batch_size * i: self.hparams.batch_size * (i + 1)],
                batch_first=True)
            )
            step_input_lens.append(input_lens[self.hparams.batch_size * i: self.hparams.batch_size * (i + 1)])
            step_labels_lens.append(labels_lens[self.hparams.batch_size * i: self.hparams.batch_size * (i + 1)])
            step_augs.append(augs[self.hparams.batch_size * i: self.hparams.batch_size * (i + 1)])

        return step_wavs, step_input_lens, step_labels, step_labels_lens, step_augs


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
