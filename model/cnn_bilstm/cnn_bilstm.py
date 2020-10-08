# This is a pytorch implementation of biLSTM

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence

import utils
import os

import pandas as pd


class CNN_BiLSTM(nn.Module):
    """ Bidirectional long-short term memory network.

    It has a bi-LSTM as the main part, then follows a three-layer perception
    classifier. Dropout is applied to the output of the bi-LSTM and every layer
    of the perception, when the probability is set to be 0.5. Finally a log
    softmax is used to normalize the output.
    """
    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
        self.args = args
        self.embed = nn.Embedding.from_pretrained(
            utils.embed_from_pretrained(args),
            freeze=self.args.freeze_embed
        )
        # self.conv1 = nn.Conv2d(in_channels=1,
        #                        out_channels=self.args.cnn_out_channels,
        #                        kernel_size=self.args.kernel_size,
        #                        stride=self.args.cnn_stride)
        self.conv1 = nn.Conv1d(in_channels=self.args.embed_num,
                               out_channels=self.args.cnn_out_channels,
                               kernel_size=self.args.kernel_size,
                               stride=self.args.cnn_stride,
                               bias=False)
        self.conv2 = nn.Conv1d(in_channels=self.args.embed_num,
                               out_channels=self.args.cnn_out_channels,
                               kernel_size=self.args.kernel_size,
                               stride=self.args.cnn_stride,
                               bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.batchnorm_forward = nn.BatchNorm1d(self.args.cnn_out_channels)
        self.batchnorm_backward = nn.BatchNorm1d(self.args.cnn_out_channels)
        self.lstm_forward = nn.LSTM(self.args.cnn_out_channels,
                                    self.args.hidden_dim,
                                    num_layers=1,
                                    dropout=self.args.dropout,
                                    bias=True)
        self.lstm_backward = nn.LSTM(self.args.cnn_out_channels,
                                     self.args.hidden_dim,
                                     num_layers=1,
                                     dropout=self.args.dropout,
                                     bias=True)
        self.fc1 = nn.Linear(self.args.hidden_dim * 2,
                             self.args.hidden_dim)
        self.fc2 = nn.Linear(self.args.hidden_dim,
                             1)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, x):
        """ Overwrites the function for forward propagation.

        It accepts a list containing 1-d tensors that contains indexes to the
        embeddings.
        """
        code, code_rev = x

        code_embedded = pad_sequence(
            [self.embed(i.to(self.args.device)) for i in code])
        code_rev_embedded = pad_sequence(
            [self.embed(i.to(self.args.device)) for i in code_rev])

        code_embedded = torch.transpose(code_embedded, 0, 1)
        code_rev_embedded = torch.transpose(code_rev_embedded, 0, 1)

        code_embedded = code_embedded.view(code_embedded.shape[0],
                                           code_embedded.shape[1],
                                           code_embedded.shape[2])
        code_rev_embedded = code_rev_embedded.view(code_rev_embedded.shape[0],
                                                   code_rev_embedded.shape[1],
                                                   code_rev_embedded.shape[2])

        code_embedded = torch.transpose(code_embedded, 1, 2)
        code_rev_embedded = torch.transpose(code_rev_embedded, 1, 2)

        # 1d convolution layer
        code_embedded = self.conv2(code_embedded)
        code_rev_embedded = self.conv2(code_rev_embedded)

        # Dropout layer
        code_embedded = self.dropout(code_embedded)
        code_rev_embedded = self.dropout(code_rev_embedded)

        code_embedded = torch.transpose(code_embedded, 1, 2)
        code_rev_embedded = torch.transpose(code_rev_embedded, 1, 2)
        code_embedded = torch.transpose(code_embedded, 0, 1)
        code_rev_embedded = torch.transpose(code_rev_embedded, 0, 1)

        # bi-layer LSTM
        lstm_forward_out, _ = self.lstm_forward(code_embedded)
        lstm_backward_out, _ = self.lstm_backward(code_rev_embedded)

        # # Use the maxpool to get the output
        # lstm_forward_out = torch.transpose(lstm_forward_out, 0, 1)
        # lstm_forward_out = torch.transpose(lstm_forward_out, 1, 2)
        # lstm_forward_out = F.tanh(lstm_forward_out)
        # lstm_forward_out = F.max_pool1d(lstm_forward_out,
        #                                 lstm_forward_out.size(2)).squeeze(2)

        # lstm_backward_out = torch.transpose(lstm_backward_out, 0, 1)
        # lstm_backward_out = torch.transpose(lstm_backward_out, 1, 2)
        # lstm_backward_out = F.tanh(lstm_backward_out)
        # lstm_backward_out = F.max_pool1d(lstm_backward_out,
        #                                  lstm_backward_out.size(2)).squeeze(2)

        # bilstm_out = torch.cat((lstm_forward_out, lstm_backward_out), dim=1)

        # Use the last output of bilstm as output
        bilstm_out = torch.cat((lstm_forward_out, lstm_backward_out), dim=2)
        lengths = torch.tensor([i.shape[0] for i in code])
        lengths = ((lengths - self.args.kernel_size) /
                   self.args.cnn_stride + 1).to(torch.long)

        idx = (lengths - 1).view(-1, 1).expand(
            len(lengths), bilstm_out.size(2))

        # if batch is first, change 0 to 1
        idx = idx.unsqueeze(0).to(self.args.device)
        bilstm_out = bilstm_out.gather(0, idx).squeeze(0)

        bilstm_out = self.dropout(bilstm_out)
        y = F.relu(self.fc1(bilstm_out))
        y = self.fc2(y)

        return y

    def initialize(self):
        """ Initializes the parameters of the model itself.

        It initializes the weights of bilstm orthogonally, which is suggested
        by a lot of blogger on the Internet, and initializes the biases of the
        bilstm with zeros.

        Args:
            None

        Returns:
            None
        """
        # Initialize gates and hidden weights of layer 1 with orthogonal
        # normalization
        print(self.lstm_forward.bias_ih_l0)
        nn.init.orthogonal_(self.lstm_forward.weight_ih_l0)
        nn.init.orthogonal_(self.lstm_forward.weight_hh_l0)
        nn.init.orthogonal_(self.lstm_backward.weight_ih_l0)
        nn.init.orthogonal_(self.lstm_backward.weight_hh_l0)

        # nn.init.xavier_normal_(self.lstm_forward.weight_ih_l0)
        # nn.init.xavier_normal_(self.lstm_forward.weight_hh_l0)
        # nn.init.xavier_normal_(self.lstm_backward.weight_ih_l0)
        # nn.init.xavier_normal_(self.lstm_backward.weight_hh_l0)

        nn.init.zeros_(self.lstm_forward.bias_ih_l0)
        nn.init.ones_(self.lstm_forward.bias_ih_l0[1])
        nn.init.zeros_(self.lstm_forward.bias_hh_l0)
        nn.init.zeros_(self.lstm_backward.bias_ih_l0)
        nn.init.ones_(self.lstm_backward.bias_ih_l0[1])
        nn.init.zeros_(self.lstm_backward.bias_hh_l0)

        nn.init.xavier_normal_(self.conv2.weight)

        # # Initialize gates and hidden weights of layer 2 with orthogonal
        # # normalization
        # nn.init.orthogonal_(self.lstm_forward.weight_ih_l1)
        # nn.init.orthogonal_(self.lstm_forward.weight_hh_l1)
        # nn.init.orthogonal_(self.lstm_backward.weight_ih_l1)
        # nn.init.orthogonal_(self.lstm_backward.weight_hh_l1)

        # # nn.init.xavier_normal_(self.bilstm.weight_ih_l0)
        # # nn.init.xavier_normal_(self.bilstm.weight_hh_l0)
        # # nn.init.xavier_normal_(self.bilstm.weight_ih_l0)
        # # nn.init.xavier_normal_(self.bilstm.weight_hh_l0)

        # nn.init.zeros_(self.lstm_forward.bias_ih_l1)
        # nn.init.ones_(self.lstm_forward.bias_ih_l1[1])
        # nn.init.zeros_(self.lstm_forward.bias_hh_l1)
        # nn.init.zeros_(self.lstm_backward.bias_ih_l1)
        # nn.init.ones_(self.lstm_backward.bias_ih_l1[1])
        # nn.init.zeros_(self.lstm_backward.bias_hh_l1)

        pass
