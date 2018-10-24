from global_config import *
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networks.attention import AttentionMask
from .networks.seq_encoder import RNNSeqEncoder
from .networks.seq_decoder import RNNSeqDecoder
class CaptionGenerator(nn.Module):
    """
    generate caption sentence given video and temporal segment
    """
    def __init__(self, hidden_dim, rnn_layer, rnn_cell, rnn_dropout, bidirectional, attention_type, context_type, scale,
                 sent_vocab_size, sent_embedding_dim, video_feature_dim, video_use_residual, max_cap_length):
        """
        :param context_type: c, cl, clr
        :param scale: scalar, used in the soft_mask_function
        """
        super(CaptionGenerator, self).__init__()
        assert bidirectional == False, "We do not support bidirectional RNN for model structure."

        # video encoder
        self.video_encoder = RNNSeqEncoder(video_feature_dim, hidden_dim, rnn_cell, rnn_layer,
                                           bidirectional, rnn_dropout, video_use_residual)

        # resolved hidden for attention layer
        resolved_hidden_dim = hidden_dim*rnn_layer*(2 if bidirectional else 1)

        # soft_mask attention module
        attention = AttentionMask(hidden_dim, resolved_hidden_dim, attention_type, context_type, scale)

        # caption generator
        self.decoder = RNNSeqDecoder(hidden_dim + hidden_dim*len(context_type), hidden_dim, sent_vocab_size,
                                     sent_embedding_dim, attention, max_cap_length, rnn_cell, rnn_layer, rnn_dropout)

    def forward(self, video_feat, video_length, video_mask, temp_seg, seg_gather_idx, sent=None, beam_size=3):
        """
        :param video_feat:      (batch_video, length_video, feature_dim)
        :param video_length:    (batch_video, 2)
        :param video_mask:      (batch_video, length_video, 1)
        :param temp_seg:        (batch_seg, 2)  normalized to (0, 1) in cw format
        :param sent:            (batch_sent, length_sent)
        :param sent_length:     (batch_sent)
        :param sent_mask:       (batch_sent, length_sent)
        :param sent_gather_idx: (batch_sent)
        :return:
        """

        # feature encoding
        video_feature, video_hidden = self.video_encoder(video_feat)

        # convert batch video to batch caption
        video_feature = video_feature.index_select(dim=0, index=seg_gather_idx)
        video_hidden = video_hidden.index_select(dim=1, index=seg_gather_idx)
        video_seq_len, _ = video_length.index_select(dim=0, index=seg_gather_idx).chunk(2, dim=1)
        video_seq_len = video_seq_len.contiguous()  # batch, 1
        video_mask = video_mask.index_select(dim=0, index=seg_gather_idx)

        # select decoder initial hidden state
        end_index = self.end_index_extractor(temp_seg, video_seq_len)  # non-differential operation!!!
        sz0, sz1, sz2, sz3 = video_hidden.size()
        gather_index = end_index.view(1, -1, 1, 1).expand(sz0, sz1, 1, sz3)
        decoder_init_hidden = torch.gather(video_hidden, dim=2, index=gather_index).squeeze(2)

        # decoding
        #video_feature = video_hidden[1, :, :, :]

        sent_prob, sent_pred, sent_len, sent_mask = self.decoder(video_feature, decoder_init_hidden, video_mask,
                                                                 temp_seg * video_seq_len, sent, beam_size)

        return sent_prob, sent_pred, sent_len, sent_mask

    def get_parameter_group(self, params):
        return [
            {'name': 'default',
             'params': self.parameters(),
             'lr': params['lr']
            },
        ]

    def end_index_extractor(self, temp_seg, video_seq_len):
        """
        :param temp_seg:       (batch, 2)
        :param video_seq_len:  (batch, 1)
        :return:
            (batch)
        """
        c, w = temp_seg.chunk(2, dim=1)  # batch, 1
        e = c + w / 2
        e, _ = torch.cat([e, torch.ones_like(e) - DELTA], dim=1).min(1)

        return (e * video_seq_len.squeeze(1)).long()

    def build_loss(self, caption_prob, ref_caption, caption_mask):
        """
        :param caption_prob: (batch_size, length, vocab_size)
        :param ref_caption:  (batch_size, length)
        :param caption_mask: (batch_size, length, 1)
        :return:
            loss
        """
        assert caption_prob.size(1) == ref_caption.size(1)
        prob = caption_prob.gather(dim=2, index=ref_caption.unsqueeze(2))  # batch_size, length, 1

        return - (prob * caption_mask).sum() / prob.size(0)
        return - (prob * caption_mask).sum() / caption_mask.sum()
        return - (prob * caption_mask).sum() / caption_prob.size(0)
