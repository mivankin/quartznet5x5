from pathlib import Path

import pytest
import torch

from src.datamodules.datamodule import ASRDataModule


def test_datamodule(batch_size):

    dm = ASRDataModule()
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    step_wavs, step_input_lens, step_labels, step_labels_lens, step_augs = batch
    assert len(step_wavs) == batch_size
    assert len(step_input_lens) == batch_size
    assert len(step_labels) == batch_size
    assert len(step_labels_lens) == batch_size
    assert len(step_augs) == batch_size
