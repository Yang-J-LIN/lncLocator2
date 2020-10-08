# This is a pytorch implementation of biLSTM

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
import sys
import utils
import os

import pandas as pd


class BiLSTM_Kernel(nn.Module):
    def __init__(self, args):
        super(BiLSTM_Kernel, self).__init__()
        self.args = args
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
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, code):
        """ Overwrites the function for forward propagation.

        It accepts a list containing 1-d tensors that contains indexes to the
        embeddings.
        """
        code_embedded, code_rev_embedded = code

        lstm_forward_out, _ = self.lstm_forward(code_embedded)
        lstm_backward_out, _ = self.lstm_backward(code_rev_embedded)

        # Use the last output of bilstm as output
        bilstm_out = torch.cat((lstm_forward_out, lstm_backward_out), dim=2)

        # print(bilstm_out[-1, 0, :])

        return bilstm_out

    def initialize(self):
        # *************** NOTICE *************** #
        # This function should be improved for multi-layer lstm
        # ***************** END **************** #
        nn.init.orthogonal_(self.lstm_forward.weight_ih_l0)
        nn.init.orthogonal_(self.lstm_forward.weight_hh_l0)
        nn.init.orthogonal_(self.lstm_backward.weight_ih_l0)
        nn.init.orthogonal_(self.lstm_backward.weight_hh_l0)


        print(self.lstm_forward.bias_ih_l0)
        print(self.lstm_forward.bias_hh_l0)

        nn.init.zeros_(self.lstm_forward.bias_ih_l0)
        # for i in range(self.args.hidden_dim):
        #     nn.init.ones_(self.lstm_forward.bias_ih_l0[1 + 4 * i])
        nn.init.ones_(self.lstm_forward.bias_ih_l0[self.args.hidden_dim:self.args.hidden_dim*2])
        nn.init.zeros_(self.lstm_forward.bias_hh_l0)
        nn.init.ones_(self.lstm_forward.bias_hh_l0[self.args.hidden_dim:self.args.hidden_dim*2])

        nn.init.zeros_(self.lstm_backward.bias_ih_l0)
        # for i in range(self.args.hidden_dim):
        #     nn.init.ones_(self.lstm_backward.bias_ih_l0[1 + 4 * i])
        nn.init.ones_(self.lstm_backward.bias_ih_l0[self.args.hidden_dim:self.args.hidden_dim*2])
        nn.init.zeros_(self.lstm_backward.bias_hh_l0)
        nn.init.ones_(self.lstm_backward.bias_hh_l0[self.args.hidden_dim:self.args.hidden_dim*2])

        print(self.lstm_forward.bias_ih_l0)
        print(self.lstm_forward.bias_hh_l0)


        # nn.init.orthogonal_(self.lstm_forward.weight_ih_l1)
        # nn.init.orthogonal_(self.lstm_forward.weight_hh_l1)
        # nn.init.orthogonal_(self.lstm_backward.weight_ih_l1)
        # nn.init.orthogonal_(self.lstm_backward.weight_hh_l1)

        # nn.init.zeros_(self.lstm_forward.bias_ih_l1)
        # nn.init.ones_(self.lstm_forward.bias_ih_l1[1])
        # nn.init.zeros_(self.lstm_forward.bias_hh_l1)
        # nn.init.zeros_(self.lstm_backward.bias_ih_l1)
        # nn.init.ones_(self.lstm_backward.bias_ih_l1[1])
        # nn.init.zeros_(self.lstm_backward.bias_hh_l1)

        # nn.init.orthogonal_(self.lstm_forward.weight_ih_l2)
        # nn.init.orthogonal_(self.lstm_forward.weight_hh_l2)
        # nn.init.orthogonal_(self.lstm_backward.weight_ih_l2)
        # nn.init.orthogonal_(self.lstm_backward.weight_hh_l2)

        # nn.init.zeros_(self.lstm_forward.bias_ih_l2)
        # nn.init.ones_(self.lstm_forward.bias_ih_l2[1])
        # nn.init.zeros_(self.lstm_forward.bias_hh_l2)
        # nn.init.zeros_(self.lstm_backward.bias_ih_l2)
        # nn.init.ones_(self.lstm_backward.bias_ih_l2[1])
        # nn.init.zeros_(self.lstm_backward.bias_hh_l2)


class CNN_2d_Kernel(nn.Module):
    def __init__(self, args):
        super(CNN_2d_Kernel, self).__init__()
        self.args = args
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=self.args.cnn_out_channels,
                              kernel_size=(self.args.embed_num,
                                           self.args.kernel_size),
                              stride=self.args.cnn_stride)
        self.conv2 = nn.Conv2d(in_channels=1,
                               out_channels=self.args.cnn_out_channels,
                               kernel_size=(self.args.cnn_out_channels,
                                            self.args.kernel_size),
                               stride=self.args.cnn_stride)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, embedded):
        embedded = torch.transpose(embedded, 0, 1)
        embedded = embedded.unsqueeze(0)
        embedded = embedded.unsqueeze(0)
        embedded = self.conv(embedded)
        embedded = torch.transpose(embedded, 1, 2)
        embedded = torch.transpose(embedded, 2, 3)
        embedded = embedded.squeeze(0)
        embedded = embedded.squeeze(0)

        # Convolution layer
        # embedded = torch.transpose(embedded, 0, 1)
        # embedded = torch.transpose(embedded, 1, 2)
        # embedded = embedded.unsqueeze(1)

        # embedded = self.conv(embedded)

        # embedded = embedded.squeeze(2)
        # embedded = torch.transpose(embedded, 1, 2)
        # embedded = torch.transpose(embedded, 0, 1)
        # Convolution layer 2
        # embedded = self.dropout(embedded)

        # embedded = torch.transpose(embedded, 0, 1)
        # embedded = torch.transpose(embedded, 1, 2)
        # embedded = embedded.unsqueeze(1)

        # embedded = self.conv2(embedded)

        # embedded = embedded.squeeze(2)
        # embedded = torch.transpose(embedded, 1, 2)
        # embedded = torch.transpose(embedded, 0, 1)

        return embedded


class CNN_1d_Kernel(nn.Module):
    def __init__(self, args):
        super(CNN_1d_Kernel, self).__init__()
        self.args = args
        self.conv = nn.Conv1d(in_channels=self.args.embed_num,
                              out_channels=self.args.cnn_out_channels,
                              kernel_size=self.args.kernel_size,
                              stride=self.args.cnn_stride,
                              bias=False)

    def forward(self, embedded):
        # Transform the structure of the embedded for 1d CNN
        embedded = torch.transpose(embedded, 0, 1)
        embedded = torch.transpose(embedded, 1, 2)

        # Convolution layer
        embedded = self.conv(embedded)

        # Transform the structure of the embedded back
        embedded = torch.transpose(embedded, 1, 2)
        embedded = torch.transpose(embedded, 0, 1)

        return embedded


class CNN_BiLSTM_Kernel(nn.Module):
    def __init__(self, args):
        super(CNN_BiLSTM_Kernel, self).__init__()
        self.args = args
        self.bilstm = BiLSTM_Kernel(args)
        self.cnn = CNN_2d_Kernel(args)
        # self.cnn = CNN_1d_Kernel(args)
        self.cnn_rev = CNN_2d_Kernel(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, embedded):
        features = [self.cnn(self.dropout(i)) for i in embedded]
        features_rev = [torch.flip(i, dims=[0]) for i in features]

        features = pad_sequence(features)
        features_rev = pad_sequence(features_rev)

        bilstm_out = self.bilstm((features, features_rev))

        # embedded_rev = [torch.flip(i, dims=[0]) for i in embedded]

        # Get the indexes of last output of reccurent neural network
        # lengths = torch.tensor([i.shape[0] for i in embedded])
        # lengths = ((lengths - self.args.kernel_size) /
        #            self.args.cnn_stride + 1).to(torch.long)
        
        lengths = [int((i.shape[0] - self.args.kernel_size) /
                   self.args.cnn_stride) for i in embedded]

        # embedded = pad_sequence(embedded)
        # embedded_rev = pad_sequence(embedded_rev)

        # # Dropout after embedding
        # embedded = self.dropout(embedded)
        # embedded_rev = self.dropout(embedded_rev)

        # # Convolutional neural network
        # embedded = self.cnn(embedded)
        # embedded_rev = self.cnn_rev(embedded_rev)

        # # Dropout after convolutional neural networks
        # # embedded = self.dropout(embedded)
        # # embedded_rev = self.dropout(embedded_rev)

        # # Recurrent neural network
        # bilstm_out = self.bilstm((embedded, embedded_rev))

        # index = (lengths - 1).view(-1, 1).expand(
        #     len(lengths), bilstm_out.size(2))
        # index = index.unsqueeze(0).to(self.args.device)

        # bilstm_out = bilstm_out.gather(0, index).squeeze(0)

        # batch_index = [i for i in range(len(lengths))]

        # output = bilstm_out[lengths, batch_index, :]

        output = torch.sum(bilstm_out, dim=0)
        output = output.transpose(0, 1)
        output = output / torch.tensor(lengths).to(self.args.device)
        output = output.transpose(0, 1)

        return output

    def initialize(self):
        self.bilstm.initialize()


class CNN_BiLSTM(nn.Module):
    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM_Kernel(args)

        if args.onehot is False:
            self.embedding = nn.Embedding.from_pretrained(
                utils.embed_from_pretrained(args),
                freeze=self.args.freeze_embed
            )
        else:
            self.embedding = nn.Embedding.from_pretrained(
                torch.eye(args.embed_num)
            )

    def forward(self, seqs):
        embedded = [self.embedding(i.to(self.args.device)) for i in seqs]
        output = self.cnn_bilstm(embedded)
        return output

    def initialize(self):
        self.cnn_bilstm.initialize()


class Linear_Classifier(nn.Module):
    def __init__(self, args):
        super(Linear_Classifier, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(2 * args.hidden_dim,
                             args.hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim,
        #                      args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim,
                             args.hidden_dim // 2)
        self.fc4 = nn.Linear(args.hidden_dim // 2,
                             1)
        self.dropout = nn.Dropout(p=self.args.dropout_LC)

    def forward(self, features):
        output = F.relu(self.fc1(features))
        # output = self.dropout(output)
        # output = F.relu(self.fc2(output))
        output = self.dropout(output)
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output


class Bitask_learning(nn.Module):
    def __init__(self, args):
        super(Bitask_learning, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM(args)
        self.classifier_1 = Linear_Classifier(args)
        self.classifier_2 = Linear_Classifier(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seqs):
        features = self.cnn_bilstm(seqs)
        features = self.dropout(features)
        pred_1 = self.classifier_1(features).squeeze()
        pred_2 = self.classifier_2(features).squeeze()
        return pred_1, pred_2

    def initialize(self):
        self.cnn_bilstm.initialize()


class Biomart_learning(nn.Module):
    def __init__(self, args):
        super(Biomart_learning, self).__init__()
        self.args = args

        self.cnn_bilstm_cds = CNN_BiLSTM(args)
        self.cnn_bilstm_3_utr = CNN_BiLSTM(args)
        self.cnn_bilstm_5_utr = CNN_BiLSTM(args)

        self.classifier_cds = Linear_Classifier(args)
        self.classifier_3_utr = Linear_Classifier(args)
        self.classifier_5_utr = Linear_Classifier(args)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seqs):
        cds, utr_3, utr_5 = seqs


        features_cds = self.cnn_bilstm_cds(cds)
        features_cds = self.dropout(features_cds)
        pred_cds = self.classifier_cds(features_cds)

        features_3_utr = self.cnn_bilstm_3_utr(utr_3)
        features_3_utr = self.dropout(features_3_utr)
        pred_3_utr = self.classifier_3_utr(features_3_utr)

        features_5_utr = self.cnn_bilstm_5_utr(utr_5)
        features_5_utr = self.dropout(features_5_utr)
        pred_5_utr = self.classifier_5_utr(features_5_utr)

        pred = pred_cds + pred_3_utr + pred_5_utr
        return pred

    def initialize(self):
        self.cnn_bilstm_cds.initialize()
        self.cnn_bilstm_3_utr.initialize()
        self.cnn_bilstm_5_utr.initialize()


class K562_learning(nn.Module):
    def __init__(self, args):
        super(K562_learning, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM(args)
        self.CNRCI_regression = Linear_Classifier(args)
        self.RCIc_regression = Linear_Classifier(args)
        self.RCIin_regression = Linear_Classifier(args)
        self.RCImem_regression = Linear_Classifier(args)
        self.RCIno_regression = Linear_Classifier(args)
        self.RCInp_regression = Linear_Classifier(args)
        self.is_protein = Linear_Classifier(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seqs):
        features = self.cnn_bilstm(seqs)
        features = self.dropout(features)
        CNRCI = self.CNRCI_regression(features)
        RCIc = self.RCIc_regression(features)
        RCIin = self.RCIin_regression(features)
        RCImem = self.RCImem_regression(features)
        RCIno = self.RCIno_regression(features)
        RCInp = self.RCInp_regression(features)
        is_protein = self.is_protein(features)

        RCI = torch.cat([CNRCI,
                         RCIin,
                         RCImem,
                         RCIc,
                         RCIno,
                         RCInp,
                         is_protein], dim=1)

        return RCI

    def initialize(self):
        self.cnn_bilstm.initialize()


class All_Celline(nn.Module):
    def __init__(self, args):
        super(All_Celline, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM(args)
        self.K562 = Linear_Classifier(args)
        self.A549 = Linear_Classifier(args)
        self.GM12878 = Linear_Classifier(args)
        self.H1_hESC = Linear_Classifier(args)
        self.HeLa_S3 = Linear_Classifier(args)
        self.HepG2 = Linear_Classifier(args)
        self.HT1080 = Linear_Classifier(args)
        self.HUVEC = Linear_Classifier(args)
        self.IMR_90 = Linear_Classifier(args)
        self.MCF_7 = Linear_Classifier(args)
        self.NCI_H460 = Linear_Classifier(args)
        self.NHEK = Linear_Classifier(args)
        self.SK_MEL_5 = Linear_Classifier(args)
        self.SK_N_DZ = Linear_Classifier(args)
        self.SK_N_SH = Linear_Classifier(args)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seqs):
        features = self.cnn_bilstm(seqs)
        features = self.dropout(features)

        K562 = self.K562(features)
        A549 = self.A549(features)
        GM12878 = self.GM12878(features)
        H1_hESC = self.H1_hESC(features)
        HeLa_S3 = self.HeLa_S3(features)

        HepG2 = self.HepG2(features)
        HT1080 = self.HT1080(features)
        HUVEC = self.HUVEC(features)
        IMR_90 = self.IMR_90(features)
        MCF_7 = self.MCF_7(features)

        NCI_H460 = self.NCI_H460(features)
        NHEK = self.NHEK(features)
        SK_MEL_5 = self.SK_MEL_5(features)
        SK_N_DZ = self.SK_N_DZ(features)
        SK_N_SH = self.SK_N_SH(features)


        cellline_RCI = torch.cat(
            [
                K562,
                A549,
                GM12878,
                H1_hESC,
                HeLa_S3,
                HepG2,
                HT1080,
                HUVEC,
                IMR_90,
                MCF_7,
                NCI_H460,
                NHEK,
                SK_MEL_5,
                SK_N_DZ,
                SK_N_SH
            ]
            , dim=1)

        return cellline_RCI

    def initialize(self):
        self.cnn_bilstm.initialize()


class Kmer_Fitter(nn.Module):
    def __init__(self, args):
        super(Kmer_Fitter, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM(args)

        self.layer_1 = nn.Linear(self.args.hidden_dim * 2,
                                 128)
        self.layer_2 = nn.Linear(128,
                                 256)

        self.dropout = nn.Dropout(args.dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self, seqs):
        features = self.cnn_bilstm(seqs)
        features = self.dropout(features)
        features = self.layer_1(features)
        features = self.layer_2(features)
        features = self.sigmoid(features)
        sums = torch.sum(features, dim=1)

        features = features.transpose(0, 1)
        features = features / sums
        features = features.transpose(0, 1)

        return features

    def initialize(self):
        self.cnn_bilstm.initialize()


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.hidden_dim * 2, 1)
        self.W_K = nn.Linear(args.hidden_dim * 2, 32)
        self.W_V = nn.Linear(args.hidden_dim * 2, 32)


class CNN_BiLSTM_Attn(nn.Module):
    def __init__(self, args):
        super(CNN_BiLSTM_Attn, self).__init__()
        self.args = args
        self.cnn_bilstm = CNN_BiLSTM(args)