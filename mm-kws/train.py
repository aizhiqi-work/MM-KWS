import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloaders.libriphrase_train import LibriPhrase_Train_Dataset, train_collate_fn
from dataloaders.libriphrase_test import LibriPhrase_Test_Dataset, collate_fn
from models import MMKWS

def compute_eer(label, pred):
    fpr, tpr, threshold = roc_curve(label, pred)
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
    

class MMKWS_Wrapper(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MMKWS()
        self.criterion = nn.BCEWithLogitsLoss()
        self.test_preds = []
        self.test_labels = []

    
    def training_step(self, batch, batch_idx):
        fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed, label, pl, mask_pl, tl, mask_tl = batch
        preds, phoneme_preds, text_preds = self.model(fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed)
        phoneme_preds = phoneme_preds.squeeze(dim=2)
        text_preds = text_preds.squeeze(dim=2)
        preds = preds.squeeze(dim=1)  # Output is [Batch size, 1], but we want [Batch size]
        phoneme_loss = self.sequence_bce_loss(phoneme_preds, pl.float(), mask_pl)
        text_loss = self.sequence_bce_loss(text_preds, tl.float(), mask_tl)
        utt_loss = self.criterion(preds, label.float())
        all_loss = utt_loss + phoneme_loss + text_loss
        self.log('train/all_loss', all_loss, on_step=True, prog_bar=True)
        self.log('train/utt_loss', utt_loss, on_step=True, prog_bar=True)
        self.log('train/text_loss', text_loss, on_step=True, prog_bar=True)
        self.log('train/phoneme_loss', phoneme_loss, on_step=True, prog_bar=True)
        return all_loss
        
    def validation_step(self, batch, batch_idx):
        fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed, label = batch
        preds, _, _ = self.model(fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed)
        preds = torch.sigmoid(preds)
        preds = preds.squeeze(dim=1)
        self.test_preds.append(preds)
        self.test_labels.append(label)


    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            eer_loss = EER()
            all_preds = torch.cat(self.test_preds)
            all_labels = torch.cat(self.test_labels)
            y_true = all_labels.cpu().detach().numpy()
            y_scores = all_preds.cpu().detach().numpy()
            # 计算 AUC
            auc = roc_auc_score(y_true, y_scores)
            eer = eer_loss(y_true, y_scores)
            self.log('test/test_auc', auc)
            self.log('test/test_eer', eer)
            self.test_preds.clear()
            self.test_labels.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, gamma=0.95),
                "frequency": 1,
                "interval": 'epoch',
            },
        }
    
    def sequence_bce_loss(self, preds, labels, mask):
        # 将预测值、标签和掩码都展平为一维向量
        preds_flat = preds.view(-1)
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)
        # 仅考虑掩码为1的位置计算二元交叉熵损失
        valid_indices = torch.where(mask_flat == 1)[0]
        valid_preds = preds_flat[valid_indices]
        valid_labels = labels_flat[valid_indices]
        # 使用PyTorch内置的二元交叉熵损失函数
        loss = F.binary_cross_entropy_with_logits(valid_preds, valid_labels)
        return loss


# 3. 设置 Trainer 和训练
if __name__ == "__main__":
    pl.seed_everything(2024)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'
    train_dataset = LibriPhrase_Train_Dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=train_collate_fn, shuffle=True, num_workers=16, drop_last=True)
    test_dataset = LibriPhrase_Test_Dataset(types='hard')
    test_dataloader = DataLoader(test_dataset, batch_size=256, collate_fn=collate_fn, shuffle=False, num_workers=8, drop_last=True)
    model = MMKWS_Wrapper()
    model_checkpoint = ModelCheckpoint(
        dirpath="/nvme01/aizq/mmkws/mmkws_submits/MMKWS_EN_Base+/logs/MMKWS+/ckpts",
        filename='epoch{epoch:02d}',
        save_top_k=-1,
    )
    logger = pl.loggers.TensorBoardLogger('/nvme01/aizq/mmkws/mmkws_submits/MMKWS_EN_Base+/logs/', name='MMKWS+')
    trainer = Trainer(devices=4, accelerator='gpu',  # strategy='ddp_find_unused_parameters_true', 
                      logger=logger, max_epochs=100, callbacks=[model_checkpoint], accumulate_grad_batches=4, precision='16-mixed')  # 设置训练器参数
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
