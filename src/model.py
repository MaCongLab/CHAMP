import torch
from torch import nn


import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
# from torch_geometric.nn import aggr
# from RelGAT_new import GATConv
from transformers import AutoModelForMaskedLM
import numpy as np
import torch.nn.functional as F

class Prot_model(torch.nn.Module):

    def __init__(self,aac_emb_dim,class_num,esm_model_path='ESMplusplus_small'):
        super().__init__()
        self.activation = nn.ELU(0.5)
        device = 'cuda'
        self.aac_emb = nn.Embedding(55,padding_idx=1,embedding_dim=aac_emb_dim)
        self.esm_model = AutoModelForMaskedLM.from_pretrained(esm_model_path,trust_remote_code=True,local_files_only=True)

        self.seq_1pcnn_kernel1 = nn.Sequential(
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim * 2, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim * 2),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim * 2, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
        )

        self.seq_1pcnn_kernel2 = nn.Sequential(
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim*2, kernel_size=2),
            nn.BatchNorm1d(aac_emb_dim*2),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim*2, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
        )

        self.seq_1pcnn_kernel3 = nn.Sequential(
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim * 2, kernel_size=3),
            nn.BatchNorm1d(aac_emb_dim * 2),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim * 2, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
        )

        self.seq_1pcnn_kernel4 = nn.Sequential(
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim * 2, kernel_size=4),
            nn.BatchNorm1d(aac_emb_dim * 2),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim * 2, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=aac_emb_dim, out_channels=aac_emb_dim, kernel_size=1),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
        )

        self.out_ln = nn.Sequential(
            nn.Linear(aac_emb_dim*4+960,aac_emb_dim*2),
            nn.BatchNorm1d(aac_emb_dim*2),
            self.activation,
            nn.Linear(aac_emb_dim*2, aac_emb_dim),
            nn.BatchNorm1d(aac_emb_dim),
            self.activation,
            nn.Linear(aac_emb_dim, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Linear(128, class_num),
        )



    def forward(self,seq_batch,attention_mask,seq_cnn):
        seq_feat = self.aac_emb(seq_cnn)
        seq_feat = seq_feat.permute(0,2,1)
        # print(seq_feat.shape)

        seq_feat_kernel1 = self.seq_1pcnn_kernel1(seq_feat)
        seq_feat_kernel1 = torch.squeeze(seq_feat_kernel1, -1)
        seq_feat_kernel1 = seq_feat_kernel1.permute(0, 2, 1)
        seq_feat_kernel1 = torch.max(seq_feat_kernel1, dim=1)[0]

        seq_feat_kernel2 = self.seq_1pcnn_kernel2(seq_feat)
        seq_feat_kernel2 = torch.squeeze(seq_feat_kernel2,-1)
        seq_feat_kernel2 = seq_feat_kernel2.permute(0,2,1)
        seq_feat_kernel2 = torch.max(seq_feat_kernel2,dim=1)[0]

        seq_feat_kernel3 = self.seq_1pcnn_kernel3(seq_feat)
        seq_feat_kernel3 = torch.squeeze(seq_feat_kernel3, -1)
        seq_feat_kernel3 = seq_feat_kernel3.permute(0, 2, 1)
        seq_feat_kernel3 = torch.max(seq_feat_kernel3, dim=1)[0]

        seq_feat_kernel4 = self.seq_1pcnn_kernel4(seq_feat)
        seq_feat_kernel4 = torch.squeeze(seq_feat_kernel4, -1)
        seq_feat_kernel4 = seq_feat_kernel4.permute(0, 2, 1)
        seq_feat_kernel4 = torch.max(seq_feat_kernel4, dim=1)[0]
        # print(total_h.shape)
        seq_feat_esm = self.esm_model(seq_batch,attention_mask)
        seq_feat_esm = seq_feat_esm.last_hidden_state
        seq_feat_esm = seq_feat_esm[:,0,:]
        total_h = torch.concat([seq_feat_kernel1,seq_feat_kernel2,seq_feat_kernel3,seq_feat_kernel4,seq_feat_esm],dim=-1)
        out_final = self.out_ln(total_h)
        return out_final
        # return None
