import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

from conformer.conformer.model_def import Conformer

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TEXT_Fusion_transformer_encoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            nlayers, 
            nhead, 
            dim_feedforward,
            dropout=0.1
        ):
        super().__init__()
        self.position_audio = PositionalEmbedding(d_model=128)
        self.position_text_g2p = PositionalEmbedding(d_model=128)
        self.position_text_lm = PositionalEmbedding(d_model=128)
        self.modality = nn.Embedding(4, 128, padding_idx=0)  # 1 for audio, 2 for g2p, 3 for text lm
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self,audio_embedding, g2p_embedding, lm_embedding):
        position_audio_encoding = self.position_audio(audio_embedding)
        position_g2p_encoding = self.position_text_g2p(g2p_embedding)
        position_lm_encoding = self.position_text_lm(lm_embedding)

        modality_audio = self.modality(1 * torch.ones((position_audio_encoding.size(0), audio_embedding.shape[1]), dtype=torch.int).to(audio_embedding.device))
        modality_g2p = self.modality(2 * torch.ones((position_g2p_encoding.size(0),  g2p_embedding.shape[1]), dtype=torch.int).to(g2p_embedding.device))
        modality_lm = self.modality(3 * torch.ones((position_lm_encoding.size(0), lm_embedding.shape[1]), dtype=torch.int).to(lm_embedding.device))

        audio_tokens = audio_embedding + position_audio_encoding + modality_audio
        g2p_tokens = g2p_embedding + position_g2p_encoding + modality_g2p
        lm_tokens = lm_embedding + position_lm_encoding + modality_lm 

        #(3) concat tokens
        input_tokens = torch.cat((audio_tokens, g2p_tokens, lm_tokens), dim=1)
        input_tokens = self.dropout(input_tokens)

        output = self.transformer_encoder(input_tokens)
        return output
    

class Audio_Fusion_transformer_encoder(nn.Module):
    def __init__(
            self, 
            d_model, 
            nlayers, 
            nhead, 
            dim_feedforward,
            dropout=0.1
        ):
        super().__init__()
        self.position_audio = PositionalEmbedding(d_model=128)
        self.position_audio_lm = PositionalEmbedding(d_model=128)
        self.modality = nn.Embedding(3, 128, padding_idx=0)  # 1 for audio, 2 for audiolm
        self.dropout = nn.Dropout(p=dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self,audio_embedding, audiolm_embedding):
        position_audio_encoding = self.position_audio(audio_embedding)
        position_audiolm_encoding = self.position_audio_lm(audiolm_embedding)

        modality_audio = self.modality(1 * torch.ones((position_audio_encoding.size(0), audio_embedding.shape[1]), dtype=torch.int).to(audio_embedding.device))
        modality_audiolm = self.modality(2 * torch.ones((position_audiolm_encoding.size(0),  audiolm_embedding.shape[1]), dtype=torch.int).to(audiolm_embedding.device))

        audio_tokens = audio_embedding + position_audio_encoding + modality_audio
        audiolm_tokens = audiolm_embedding + position_audiolm_encoding + modality_audiolm

        #(3) concat tokens
        input_tokens = torch.cat((audio_tokens, audiolm_tokens), dim=1)
        input_tokens = self.dropout(input_tokens)

        output = self.transformer_encoder(input_tokens)
        return output
    

class GRUFCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUFCModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_last_output = gru_out[:, -1, :]
        fc_out = self.fc(gru_last_output)
        return fc_out
    

import torch.nn as nn
class Projection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Projection, self).__init__()
        layers = []
        layers.append(nn.LayerNorm(input_dim)) 
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.SiLU())
        self.projection_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.projection_block(x)
    
    
class MMKWS(nn.Module):
    def __init__(self):
        super().__init__()
        self.audioencoder = Conformer(
            input_dim= 80,
            encoder_dim= 128,
            num_encoder_layers= 6,
            num_attention_heads= 4,
        )
        self.g2p_projection = Projection(input_dim=256, output_dim=128)
        self.lm_projection = Projection(input_dim=768, output_dim=128)
        self.audiolm_projection = Projection(input_dim=1024, output_dim=128)
        self.text_fusion_transformer = TEXT_Fusion_transformer_encoder(d_model=128,nlayers=2,nhead=4,dim_feedforward=512,dropout=0.1)
        self.audio_fusion_transformer = Audio_Fusion_transformer_encoder(d_model=128,nlayers=2,nhead=4,dim_feedforward=512,dropout=0.1)
        self.gru1 = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=64)
        self.gru2 = GRUFCModel(input_dim=128, hidden_dim=128, output_dim=64)
        self.fc = nn.Linear(64, 1)
        self.phoneme_fc = nn.Linear(128, 1)
        self.text_fc = nn.Linear(128, 1)


    def forward(self, fbank_feature, lengths, g2p_embed, lm_embed, audiolm_embed):
        audio_embedding = self.audioencoder(fbank_feature, lengths)[0]
        g2p_embedding = self.g2p_projection(g2p_embed)
        lm_embedding = self.lm_projection(lm_embed)
        audiolm_embedding = self.audiolm_projection(audiolm_embed)
        
        fusion_text = self.text_fusion_transformer(audio_embedding, g2p_embedding, lm_embedding)
        fusion_audio = self.audio_fusion_transformer(audio_embedding, audiolm_embedding)
        fusion = self.gru1(fusion_text)+self.gru2(fusion_audio)
        fusion_pred = self.fc(fusion)

        fusion_phoneme_pred = self.phoneme_fc(fusion_text[:, audio_embedding.shape[1]:(audio_embedding.shape[1]+g2p_embedding.shape[1]), :])
        fusion_text_pred = self.text_fc(fusion_text[:, (audio_embedding.shape[1]+g2p_embedding.shape[1]):, :])
        return fusion_pred, fusion_phoneme_pred, fusion_text_pred
