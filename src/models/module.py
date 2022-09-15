from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, CharErrorRate
from torchmetrics.classification.accuracy import Accuracy

try:
    from torchmetrics import WER as WordErrorRate
except ImportError:
    from torchmetrics import WordErrorRate

import torch_optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.encoders import TextEncDec, GreedyCTCDecoder

text_enc_dec = TextEncDec()
greedy_decoder = GreedyCTCDecoder(text_enc_dec.id_to_token)
    
class QuartznetModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        #optimizer: torch.optim.Optimizer,
        #scheduler: torch.optim.lr_scheduler,
        encoder: Any = text_enc_dec,
        greedy_decoder: Any = greedy_decoder
    ):
        super().__init__()
        
        self.save_hyperparameters(logger=False, ignore=['net'])
        
        self.net = net
        
        self.criterion = nn.CTCLoss()
        
        self.encoder = encoder
        self.greedy_decoder = greedy_decoder
        
        self.train_cer = CharErrorRate()
        self.val_cer = CharErrorRate()
        self.test_acc = CharErrorRate()
        
        self.train_wer = WordErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wcc = WordErrorRate()   
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        
        self.val_cer_best = MinMetric()
        self.val_wer_best = MinMetric()
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks        
        self.val_cer_best.reset()
        
    def step(self, batch: Any):
        wavs, answ, wavs_len, answ_len = batch
        output = self.forward(wavs)
        output = F.log_softmax(output, dim=1)
        output = output.transpose(0, 1).transpose(0, 2)
        loss = self.criterion(output, answ, wavs_len, answ_len)
        
        return loss, output
            
    def training_step(self, batch: Any, batch_idx: int):
        wavs, wavs_len, answ, answ_len, augs_type = batch
        indices = np.random.permutation(np.arange(len(wavs)))
        
        step_loss = 0
        step_preds = []
        for idx in indices:
            batch = (wavs[idx], answ[idx], wavs_len[idx], answ_len[idx])
            loss, output = self.step(batch)
            
            step_loss += loss.item()

            self.train_loss(loss)
            for j in range(len(answ[idx])):
                targets = ''.join(self.encoder.decode(answ[idx][j].cpu().numpy())).replace('-', '')
                preds = ' '.join(self.greedy_decoder(output.transpose(0, 1)[j]))

                self.train_cer(preds, targets)
                self.train_wer(preds, targets)
                self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log('train/cer', self.train_cer, on_step=True, on_epoch=True, prog_bar=True)
                self.log('train/wer', self.train_wer, on_step=True, on_epoch=True, prog_bar=True)

        return
    
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        wavs, wavs_len, answ, answ_len, augs_type = batch
        indices = np.random.permutation(np.arange(len(wavs)))
        
        step_loss = 0
        step_preds = []
        for idx in indices:
            batch = (wavs[idx], answ[idx], wavs_len[idx], answ_len[idx])
            loss, output = self.step(batch)
            
            step_loss += loss.item()

            self.val_loss(loss)
            for j in range(len(answ[idx])):
                targets = ''.join(self.encoder.decode(answ[idx][j].cpu().numpy())).replace('-', '')
                preds = ' '.join(self.greedy_decoder(output.transpose(0, 1)[j]))
                self.val_cer(preds, targets)
                self.val_wer(preds, targets)                
                self.log('val/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log('val/cer', self.train_cer, on_step=True, on_epoch=True, prog_bar=True)
                self.log('val/wer', self.train_wer, on_step=True, on_epoch=True, prog_bar=True)
		
        return
        
    
    def validation_epoch_end(self, outputs: List[Any]):
        pass
        
    def test_step(self, batch: Any, batch_idx: int):
        wavs, wavs_len, answ, answ_len, augs_type = batch
        indices = np.random.permutation(np.arange(len(wavs)))
        
        step_loss = 0
        step_preds = []
        for idx in indices:
            batch = (wavs[idx], answ[idx], wavs_len[idx], answ_len[idx])
            loss, output = self.step(batch)
            
            step_loss += loss.item()

            self.train_loss(loss)
            for j in range(len(answ[idx])):
                targets = ''.join(self.encoder.decode(answ[idx][j].cpu().numpy())).replace('-', '')
                preds = ' '.join(self.greedy_decoder(output.transpose(0, 1)[j]))
                self.test_cer(preds, targets)
                self.test_wer(preds, targets)                
                self.log('test/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
                self.log('test/cer', self.train_cer, on_step=True, on_epoch=True, prog_bar=True)
                self.log('test/wer', self.train_wer, on_step=True, on_epoch=True, prog_bar=True)
		
        return
    
    
    def test_epoch_end(self, outputs: List[Any]):
        pass
    
    def configure_optimizers(self):
        optimizer = torch_optimizer.NovoGrad(
                                self.parameters(),
                                lr=0.01,
                                betas=(0.8, 0.5),
                                weight_decay=0.001,
        )
        scheduler  = CosineAnnealingLR(optimizer,
                                       T_max=100,
                                       eta_min=0,
                                       last_epoch=-1
        )
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler
        }
    
    
if __name__ == '__main__':
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / 'configs' / 'model' / 'test/yaml')
    _ = hydra.utils.instantiate(cfg)
