# -*- coding: utf-8 -*-

from functools import total_ordering
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU


def get_graph_feature(atom_fea, nbr_fea, nbr_fea_idx, atom_fea_len):
    N, M = nbr_fea_idx.shape
    atom_nbr_fea = atom_fea[nbr_fea_idx, :]
    total_nbr_fea = torch.cat([atom_fea.unsqueeze(1).expand(N, M, atom_fea_len),
                            atom_nbr_fea, nbr_fea], dim=2)
    total_nbr_fea = total_nbr_fea.transpose(2, 1).contiguous()
    return total_nbr_fea


def get_graph_feature2(atom_fea, nbr_fea_idx, atom_fea_len):
    N, M = nbr_fea_idx.shape
    atom_nbr_fea = atom_fea[nbr_fea_idx, :]
    atom_fea_expand = atom_fea.unsqueeze(1).expand(N, M, atom_fea_len)
    relative_fea = atom_nbr_fea - atom_fea_expand
    total_nbr_fea = torch.cat([atom_fea_expand, relative_fea], dim=2)
    total_nbr_fea = total_nbr_fea.transpose(2, 1).contiguous()
    return total_nbr_fea


class TRANSFORMER(nn.Module):
    def __init__(self, args, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, output_channels=1):

        super(TRANSFORMER, self).__init__()
        self.args = args
        self.embedding = nn.Sequential(nn.Linear(orig_atom_fea_len, 32),
                                       nn.BatchNorm1d(32),
                                       nn.ReLU())
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.conv1 = nn.Sequential(nn.Conv1d(2*32+nbr_fea_len, 64, kernel_size=1),
                                   self.bn1,
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(2*64, 128, kernel_size=1),
                                   self.bn2,
                                   nn.ReLU())        
        self.conv3 = nn.Sequential(nn.Conv1d(2*192, 512, kernel_size=1),
                                   self.bn3,
                                   nn.ReLU())

        self.pt_last = Point_Transformer_Last(args)

        self.linear1 = nn.Linear(3*512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)      
        self.linear3 = nn.Linear(512, output_channels)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        N, M = nbr_fea_idx.shape
        atom_fea = self.embedding(atom_fea)
        atom_fea = get_graph_feature(atom_fea, nbr_fea, nbr_fea_idx, atom_fea_len=32)
        atom_fea = self.conv1(atom_fea)
        x1 = atom_fea.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature2(x1, nbr_fea_idx, atom_fea_len=64)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2), dim=1)
        x = get_graph_feature2(x, nbr_fea_idx, atom_fea_len=192)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x4 = self.pt_last(x3, crystal_atom_idx)
        x = self.pooling(x4, crystal_atom_idx)
    
        x = F.relu(self.bn5(self.linear1(x)))
        x = F.relu(self.bn6(self.linear2(x)))
        out = self.linear3(x)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 2, 1)
        self.k_conv = nn.Conv1d(channels, channels // 2, 1)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

        self.softmax = nn.Softmax(dim=-2)

    def forward(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
                    atom_fea.data.shape[0]
        # input_size:(b*n, c), transform (b*n, c) --> (b*n,c,1)
        N, M = atom_fea.shape
        atom_fea1 = atom_fea.view(N,M,-1)
        x_q = self.q_conv(atom_fea1)
        x_q = x_q.view(N,-1)
        x_k = self.k_conv(atom_fea1)
        x_k = x_k.view(N,-1)
        _, dk = x_k.shape
        x_v = self.v_conv(atom_fea1)
        x_v = x_v.view(N,-1)
        # # dot production 
        batch_energy = [1/math.sqrt(dk) * torch.mm(x_q[idx_map], x_k[idx_map].transpose(1, 0).contiguous()) for idx_map in crystal_atom_idx]
        
        energys = [self.softmax(energy) for energy in batch_energy]
        attention = [self.l1_norm(energy, dim=-1) for energy in energys]

        x_r = torch.cat([torch.mm(x_v[idx_map].transpose(1, 0).contiguous(), atten.t()) for idx_map, atten in zip(crystal_atom_idx, attention)], dim=1)
        x_trans = atom_fea - x_r.transpose(1, 0).contiguous()
        x_trans = x_trans.view(N,M,-1)
        x_r = self.act(self.after_norm(self.trans_conv(x_trans)))
        x = atom_fea + x_r.view(N,-1)
        return x

    def l1_norm(self, x, dim):
        x = x / (1e-9 + x.sum(dim=dim, keepdim=True))
        return x


class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=512):
        super(Point_Transformer_Last, self).__init__()
        self.args = args

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x, crystal_atom_idx):
        # B, D, N
        x1 = self.sa1(x, crystal_atom_idx)
        x2 = self.sa2(x1, crystal_atom_idx)
        x3 = self.sa3(x2, crystal_atom_idx)      
        x = torch.cat((x1, x2, x3), dim=1)
        return x
