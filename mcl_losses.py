#!/usr/bin/env python
# coding=utf-8
from torch import nn
import torch
import torch.nn.functional as F
import math

def deal_nan(M):
    inf = M.max()
    if torch.isnan(M).any():#True
        print('WARNING: ISA has nan')
        M = torch.where(torch.isnan(M), torch.full_like(M, 0.), M)
    if torch.isinf(M).any():#True
        print('WARNING: ISA has inf')
        M = torch.where(torch.isinf(M), torch.full_like(M, inf), M)

    return M

class SupMCL(nn.Module):
    def __init__(self, number_net, feat_dim, kd_T, tau):
        super(SupMCL, self).__init__()
        self.number_net = number_net
        self.feat_dim = feat_dim
        self.kl = KLDiv(T=kd_T)
        self.tau = tau

    def forward(self, embeddings, labels):
        #! why need labels?
        batchSize = embeddings[0].size(0)

        labels = labels.unsqueeze(0)
        device = labels.device
        intra_mask = torch.eq(labels, labels.T).float() - torch.eye(labels.size(1), device = labels.device)
        inter_mask = torch.eq(labels, labels.T).float()
        diag_mask = (1.-torch.eye(labels.size(1), device = labels.device))

        inter_logits = []
        soft_icl_loss = 0.
        for i in range(self.number_net):
            for j in range(i + 1, self.number_net):
                cos_simi_ij = torch.div(
                    torch.mm(embeddings[i], embeddings[j].T),
                    self.tau)
                inter_logits.append(cos_simi_ij)

                cos_simi_ji = torch.div(
                    torch.mm(embeddings[j], embeddings[i].T),
                    self.tau)
                inter_logits.append(cos_simi_ji)

                soft_icl_loss += self.kl(cos_simi_ij, cos_simi_ji.detach())
                soft_icl_loss += self.kl(cos_simi_ji, cos_simi_ij.detach())

        icl_loss = 0.
        for logit in inter_logits:
            log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            icl_loss += - mean_log_prob_pos.mean()

        
        intra_logits = []
        for i in range(self.number_net):
            cos_simi = torch.div(
                torch.mm(embeddings[i], embeddings[i].T),
                self.tau)
            intra_logits.append(cos_simi)

        soft_vcl_loss = 0.
        for i in range(self.number_net):
            for j in range(self.number_net):
                if i != j:
                    soft_vcl_loss += self.kl(intra_logits[i], intra_logits[j].detach())

        vcl_loss = 0.
        for logit in intra_logits:
            log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            vcl_loss += - mean_log_prob_pos.mean()

        return vcl_loss, soft_vcl_loss, icl_loss, soft_icl_loss

    def icl(self, embeddings, labels):
        #! why need labels?
        batchSize = embeddings[0].size(0)

        labels = labels.unsqueeze(0)
        device = labels.device
        intra_mask = torch.eq(labels, labels.T).float() - torch.eye(labels.size(1), device = labels.device)
        inter_mask = torch.eq(labels, labels.T).float()
        diag_mask = (1.-torch.eye(labels.size(1), device = labels.device))

        icl_loss = 0.
        inter_logits = []
        for i in range(self.number_net):
            for j in range(i + 1, self.number_net):
                cos_simi_ij = torch.div(
                    torch.mm(embeddings[i], embeddings[j].T),
                    self.tau)
                inter_logits.append(cos_simi_ij)

                cos_simi_ji = torch.div(
                    torch.mm(embeddings[j], embeddings[i].T),
                    self.tau)
                inter_logits.append(cos_simi_ji)


        for logit in inter_logits:
            # logit values are like 4000~
            #log_prob = logit - torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True))
            log_prob = logit - torch.log((logit * diag_mask).sum(1, keepdim=True))
            #deal_nan(torch.log((torch.exp(logit) * diag_mask).sum(1, keepdim=True)))
            mean_log_prob_pos = (intra_mask * log_prob).sum(1) / intra_mask.sum(1)
            icl_loss += - mean_log_prob_pos.mean()

        
        return icl_loss




class KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T**2)
        return loss
