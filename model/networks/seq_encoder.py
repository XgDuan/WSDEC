from global_config import *
import torch.nn as nn
from torch.autograd import Variable

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class RNNSeqEncoder(nn.Module):
    """
    sequence encoder: encode a sequence
    """
    def __init__(self, input_dim, hidden_dim, rnn_cell, n_layers, bidirectional, rnn_dropout, use_residual):
        """
        :param input_dim:
        :param hidden_dim:
        :param rnn_cell:
        :param n_layers:
        :param rnn_dropout:
        :param use_residual:
        """
        super(RNNSeqEncoder, self).__init__()
        if rnn_cell.lower() == 'rnn':
            self.rnn = nn.RNN
        elif rnn_cell.lower() == 'gru':
            self.rnn = nn.GRU
        else:
            raise Exception('other rnn_cells are not implemented currently')
        # rnn_cell
        self.rnn_cell = self.rnn(input_dim, hidden_dim, n_layers,
                                 bidirectional=bidirectional, batch_first=True, dropout=rnn_dropout)
        # residual_connection
        self.use_residual = use_residual
        if use_residual:
            if input_dim == hidden_dim:
                self.linear = Identity()
            else:
                self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
            self.forward = self._forward_residual
        else:
            self.forward = self._forward

    def forward(self, *inputs):
        raise NotImplementedError

    def _forward_residual(self, input_features):
        """
        :param input_features: (batch_size, seq_len, feature_dim)
        :return:
            output: (batch, length, hidden_dim)
            hidden: (directions*layers, batch, length, hidden_dim)
        """
        hidden_list = []
        output_list = []
        hidden = None
        self.rnn_cell.flatten_parameters()
        for i in range(input_features.size(1)):
            output, hidden = self.rnn_cell(input_features[:, i, :].unsqueeze(1), hidden)
            residual_output = self.linear(input_features[:, i, :].unsqueeze(1)) + output
            output_list.append(residual_output)
            hidden_list.append(hidden.unsqueeze(2))  # batch, 1, ~ * hidden_dim
        return torch.cat(output_list, dim=1), torch.cat(hidden_list, dim=2)

    def _forward(self, input_features):
        """
        :param input_features: (batch_size, seq_len, feature_dim)
        :return:
            output: (batch, length, hidden_dim)
            hidden: (directions*layers, batch, length, hidden_dim)
        """
        hidden_list = []
        output_list = []
        hidden = None
        self.rnn_cell.flatten_parameters()
        for i in range(input_features.size(1)):
            output, hidden = self.rnn_cell(input_features[:, i, :].unsqueeze(1), hidden)
            output_list.append(output)
            hidden_list.append(hidden.unsqueeze(2))  # batch, 1, ~ * hidden_dim
        return torch.cat(output_list, dim=1), torch.cat(hidden_list, dim=2)


class RNNSeqMaskEncoder(nn.Module):
    """
    sequence encoder: encode a sequence
    """
    def __init__(self, input_dim, hidden_dim, rnn_cell, n_layers, bidirectional, rnn_dropout, use_residual):
        """
        :param input_dim:
        :param hidden_dim:
        :param rnn_cell:
        :param n_layers:
        :param rnn_dropout:
        :param use_residual:
        """
        super(RNNSeqMaskEncoder, self).__init__()
        if rnn_cell.lower() == 'rnn':
            self.rnn = nn.RNN
        elif rnn_cell.lower() == 'gru':
            self.rnn = nn.GRU
        else:
            raise Exception('other rnn_cells are not implemented currently')
        # rnn_cell
        self.rnn_cell = self.rnn(input_dim, hidden_dim, n_layers,
                                 bidirectional=bidirectional, batch_first=True, dropout=rnn_dropout)

        self.default_hidden = Variable(torch.zeros(n_layers * (2 if bidirectional else 1), 1, hidden_dim).cuda())
        # residual_connection
        self.use_residual = use_residual
        if use_residual:
            if input_dim == hidden_dim:
                self.linear = Identity()
            else:
                self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
            self.forward = self._forward_residual
        else:
            self.forward = self._forward

    def forward(self, input_features, input_mask):
        raise NotImplementedError

    def _forward_residual(self, input_features, start_index):
        """
        we encode the input features within in the mask.
        :param input_features: (batch_size, seq_len, feature_dim)
        :param start_index: (batch_size) long, in (s, e) format
        :return:
            output: (batch, length, hidden_dim)
            hidden: (directions*layers, batch, length, hidden_dim)
        """
        hidden_list = []
        output_list = []
        default_hidden = self.default_hidden.repeat(1, input_features.size(0), 1)  # ~, batch, 1, hidden_dim
        hidden = default_hidden
        start_index = start_index.unsqueeze(0).unsqueeze(2)  # 1, batch, 1
        self.rnn_cell.flatten_parameters()
        for i in range(input_features.size(1)):
            hidden = hidden * (start_index < i).float() + default_hidden * (start_index >= i).float()
            output, hidden = self.rnn_cell(input_features[:, i, :].unsqueeze(1), hidden)
            residual_output = self.linear(input_features[:, i, :].unsqueeze(1)) + output
            output_list.append(residual_output)
            hidden_list.append(hidden.unsqueeze(2))  # ~, batch, 1,  hidden_dim
        return torch.cat(output_list, dim=1), torch.cat(hidden_list, dim=2)

    def _forward(self, input_features, input_mask):
        """
        :param input_features: (batch_size, seq_len, feature_dim)
        :return:
            output: (batch, length, hidden_dim)
            hidden: (directions*layers, batch, length, hidden_dim)
        """
        hidden_list = []
        output_list = []
        hidden = None
        self.rnn_cell.flatten_parameters()
        for i in range(input_features.size(1)):
            output, hidden = self.rnn_cell(input_features[:, i, :].unsqueeze(1), hidden)
            output_list.append(output)
            hidden_list.append(hidden.unsqueeze(2))  # batch, 1, ~ * hidden_dim
        return torch.cat(output_list, dim=1), torch.cat(hidden_list, dim=2)

# class CNNSeqEncoder(nn.Module):
#     raise NotImplementedError
