from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from dataloaders.libriphrase_test import LibriPhrase_Test_Dataset, collate_fn
import torch.nn as nn
from models import MMKWS
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import sklearn
import numpy as np
import torch.nn as nn
import torch

def compute_eer(label, pred):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr

    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    eer = (eer_1 + eer_2) / 2
    return eer

class EER(nn.Module):
    def __init__(self):
        super(EER, self).__init__()
        self.score = 0.0
        self.count = 0.0

    def forward(self, y_true, y_pred):
        label_np = y_true.flatten()  # Convert to numpy array
        pred_np = y_pred.flatten()  # Convert to numpy array

        eer_value = compute_eer(label_np, pred_np)

        self.score += eer_value
        self.count += 1

        return torch.tensor(self.score / self.count)

# 1. 定义 LightningModuley
class MMKWS_Wrapper(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MMKWS()
        self.criterion = nn.BCEWithLogitsLoss()
        self.test_preds, self.test_labels = [], []


    def test_step(self, batch, batch_size):
        fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed, label = batch
        preds, _, _ = self.model(fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed)
        preds = torch.sigmoid(preds)
        preds = preds.squeeze(dim=1)
        self.test_preds.append(preds)
        self.test_labels.append(label)
        

    def on_test_epoch_end(self):
        eer_loss = EER()
        all_preds = torch.cat(self.test_preds)
        all_labels = torch.cat(self.test_labels)
        y_true = all_labels.cpu().detach().numpy()
        y_scores = all_preds.cpu().detach().numpy()
        auc = roc_auc_score(y_true, y_scores)
        eer = eer_loss(y_true, y_scores)
        self.log('test_auc', auc)
        self.log('test_eer', eer)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        return optim


# 3. 设置 Trainer 和训练
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    test_dataset = LibriPhrase_Test_Dataset(types='easy')
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=collate_fn, shuffle=False, num_workers=24, drop_last=False)
    model = MMKWS_Wrapper.load_from_checkpoint("/nvme01/aizq/mmkws/mmkws_submits/MMKWS_EN_Base+/logs/MMKWS+/ckpts/epochepoch=19.ckpt")
    model.eval()
    trainer = Trainer(devices=1, accelerator='gpu')  # 设置训练器参数
    trainer.test(model, test_dataloader)
    pl.seed_everything(1234)
    test_dataset = LibriPhrase_Test_Dataset(types='hard')
    test_dataloader = DataLoader(test_dataset, batch_size=1024, collate_fn=collate_fn, shuffle=False, num_workers=24, drop_last=False)
    model = MMKWS_Wrapper.load_from_checkpoint("/nvme01/aizq/mmkws/mmkws_submits/MMKWS_EN_Base+/logs/MMKWS+/ckpts/epochepoch=19.ckpt")
    trainer = Trainer(devices=1, accelerator='gpu', )  # 设置训练器参数
    model.eval()
    trainer.test(model, test_dataloader)
