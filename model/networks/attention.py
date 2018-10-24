from math import sqrt

from global_config import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    """
    Attention Base Class
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=-1):
        """
        :param feature_dim: feature dimension of the input
        :param hidden_dim:  HERE, USE (hidden_dim * n_layers * n_directions) instead of your true hidden_dim
        """
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

    def forward(self, feature, hidden, mask):
        """
        :param feature: Float (batch, length, feature_dim)
        :param hidden:  Float (batch, hidden_dim), pls arrange the size of hidden to the specific format
        :param mask:    Float (batch, length, 1)
        :return:
            res:   Float (batch, feature_dim)
            alpha: Float (batch, length)
        """
        raise NotImplementedError()


class AttentionMean(Attention):
    def forward(self, feature, hidden, mask, pe_size=-1):
        feature_masked_sum = torch.sum(feature * mask, dim=1)  # batch, feature_dim
        feature_masked_weight = torch.sum(mask, dim=1) + DELTA  # batch, 1
        res = feature_masked_sum / feature_masked_weight
        return res, mask.squeeze(2)


class AttentionType0(Attention):
    """
    a_i = (f_i * W * hidden) / sqrt(d_k))
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=-1):
        super(AttentionType0, self).__init__(feature_dim, hidden_dim, pe_size)
        self.linear = nn.Linear(feature_dim, hidden_dim, bias=False)  # the matrix W: feature_dim * hidden_dim

    def forward(self, feature, hidden, mask):
        alpha= self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True) / sqrt(self.hidden_dim)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature
        return res, alpha.squeeze(2)


class AttentionType1(Attention):
    """
    a_i = (f_i * W * hidden)
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=-1):
        super(AttentionType1, self).__init__(feature_dim, hidden_dim, pe_size)
        self.linear = nn.Linear(feature_dim, hidden_dim, bias=False)  # the matrix W: feature_dim * hidden_dim

    def forward(self, feature, hidden, mask):
        alpha = self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature_dim
        return res, alpha.squeeze(2)


class AttentionType2(Attention):
    """
    a_i = (f_i * W * hidden)
    advanced position aware attention module
    """
    def __init__(self, feature_dim=None, hidden_dim=None, pe_size=100):
        super(AttentionType2, self).__init__(feature_dim, hidden_dim, pe_size)
        self.linear = nn.Linear(feature_dim, hidden_dim, bias=False)  # the matrix W: feature_dim * hidden_dim
        if pe_size != -1:
            self.pe_size = pe_size
            self.position_embedding = nn.Embedding(pe_size, feature_dim)
            nn.init.eye(self.position_embedding.weight)  # to be tested
            self.forward = self._forward_pe
        else:
            self.forward = self._forward

    def forward(self, feature, hidden, mask):
        raise NotImplementedError()

    def _forward_pe(self, feature, hidden, mask):
        seq_len = mask.sum(1)  # batch, 1
        pe_index = Variable((torch.arange(0, feature.size(1)).unsqueeze(0).repeat(feature.size(0), 1)).cuda())  # (batch, seq_len), float
        pe_index = pe_index / (seq_len + DELTA)  # (batch, seq_len), normalized to (0, 1+D)
        pe_index, _ = torch.stack([pe_index, torch.ones_like(pe_index) - DELTA], dim=2).min(2)  #  (0, 1-delta)
        pe_index = (pe_index * self.pe_size).long().detach()  # batch, seq_len
        feature = feature + self.position_embedding(pe_index)

        alpha= self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True) / sqrt(self.hidden_dim)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature
        return res, alpha.squeeze(2)


    def _forward(self, feature, hidden, mask):
        alpha = self.linear(feature)  # batch, length, hidden_dim
        alpha = alpha * (hidden.unsqueeze(1).expand_as(alpha))  # batch, length, hidden_dim
        alpha = alpha.sum(2, keepdim=True)  # batch, length, 1
        mask_helper = torch.zeros_like(alpha)
        mask_helper[mask == 0] = - float('inf')
        alpha = alpha + mask_helper
        alpha = F.softmax(alpha, dim=1)  # batch, length, 1
        res = (alpha * feature).sum(1)  # batch, feature_dim
        return res, alpha.squeeze(2)

# Attention With temporal segments

class ContextMaskC(nn.Module):
    def __init__(self, scale):
        super(ContextMaskC, self).__init__()
        self.scale = scale

    def forward(self, index, c, w):
        """
        :param index: Float (batch, length, 1)
        :param c:     Float (batch, 1, 1)
        :param w:     Float (batch, 1, 1)
        :return:
            masks: Float (batch, length, 1)
        """
        return F.sigmoid(self.scale * (index - c + w / 2)) - F.sigmoid(self.scale * (index - c - w / 2))


class ContextMaskL(nn.Module):
    def __init__(self, scale):
        super(ContextMaskL, self).__init__()
        self.scale = scale

    def forward(self, index, c, w):
        """
        :param index: Float (batch, length, 1)
        :param c:     Float (batch, 1, 1)
        :param w:     Float (batch, 1, 1)
        :return:
            masks: Float (batch, length, 1)
        """
        return F.sigmoid(- self.scale * (index - c + w / 2))


class ContextMaskR(nn.Module):
    def __init__(self, scale):
        super(ContextMaskR, self).__init__()
        self.scale = scale

    def forward(self, index, c, w):
        """
        :param index: Float (batch, length, 1)
        :param c:     Float (batch, 1, 1)
        :param w:     Float (batch, 1, 1)
        :return:
            masks: Float (batch, length, 1)
        """
        return F.sigmoid(self.scale * (index- c - w / 2))

class AttentionMask(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_type, context_type, scale):
        """
        :param feature_dim:
        :param hidden_dim:  pls arrange the size of hidden to the specific format
        :param attention_type:
        :param context_type:
        :param scale:
        """
        super(AttentionMask, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.context_type = context_type.lower()
        if self.context_type == 'c':
            self.mask_c = ContextMaskC(scale)
            self.attention_c = self._build_attention(attention_type)
            self.forward = self._forwardc
        elif self.context_type == 'cl':
            self.mask_c = ContextMaskC(scale)
            self.attention_c = self._build_attention(attention_type)
            self.mask_r = ContextMaskR(scale)
            self.attention_r = self._build_attention(attention_type)
            self.forward = self._forwardcl
        elif self.context_type == 'clr':
            self.mask_c = ContextMaskC(scale)
            self.attention_c = self._build_attention(attention_type)
            self.mask_r = ContextMaskR(scale)
            self.attention_r = self._build_attention(attention_type)
            self.mask_l = ContextMaskL(scale)
            self.attention_l = self._build_attention(attention_type)
            self.forward = self._forwardclr
        else:
            raise Exception('other attention types are not supported currently')

    def _build_attention(self, attention_type):
        if attention_type.lower() == 'mean':
            return AttentionMean(self.feature_dim, self.hidden_dim)
        elif attention_type.lower() == 'type0':
            return AttentionType0(self.feature_dim, self.hidden_dim)
        elif attention_type.lower() == 'type1':
            return AttentionType1(self.feature_dim, self.hidden_dim)
        else:
            raise Exception('other attention types are not supported currently')

    def forward(self, feature, hidden, segment, mask):
        """
        :param feature: (batch, length, feature_dim)
        :param hidden:  (batch, hidden_dim)
        :param segment: (batch, 2)
        :param mask:    (batch, 2)
        :return:
            context
        """
        raise NotImplementedError()

    def _forwardc(self, feature, hidden, segment, mask):
        batch_size, seq_len = feature.size(0), feature.size(1)
        c, w = segment.unsqueeze(2).chunk(2, dim=1)  # batch, 1, 1
        mask_index = Variable(FloatTensor(range(seq_len)).expand(batch_size, seq_len).unsqueeze(2))  # batch_size, seq_len, 1
        c_context, _ = self.attention_c(feature, hidden, mask*self.mask_c(mask_index, c, w))
        return c_context

    def _forwardcl(self, feature, hidden, segment, mask):
        batch_size, seq_len = feature.size(0), feature.size(1)
        c, w = segment.unsqueeze(2).chunk(2, dim=1)  # batch, 1, 1
        mask_index = Variable(FloatTensor(range(seq_len)).expand(batch_size, seq_len).unsqueeze(2))  # batch_size, seq_len, 1
        l_context, _ = self.attention_l(feature, hidden, mask * self.mask_l(mask_index, c, w))
        c_context, _ = self.attention_c(feature, hidden, mask * self.mask_c(mask_index, c, w))
        return torch.cat([l_context, c_context], dim=1)

    def _forwardclr(self, feature, hidden, segment, mask):
        batch_size, seq_len = feature.size(0), feature.size(1)
        c, w = segment.unsqueeze(2).chunk(2, dim=1)  # batch, 1, 1
        mask_index = Variable(FloatTensor(range(seq_len)).expand(batch_size, seq_len).unsqueeze(2))  # batch_size, seq_len, 1
        l_context, _ = self.attention_l(feature, hidden, mask * self.mask_l(mask_index, c, w))
        r_context, _ = self.attention_r(feature, hidden, mask * self.mask_r(mask_index, c, w))
        c_context, _ = self.attention_c(feature, hidden, mask * self.mask_c(mask_index, c, w))
        return torch.cat([l_context, c_context, r_context], dim=1)
