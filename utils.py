# This module inlcudes functions that will be used in many other modules.
import logging

# import word2vec
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch.nn.utils.rnn import pack_sequence
from sklearn import metrics

logging.basicConfig(filename="train.log",
                    filemode="a",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.DEBUG)


def tokenize_seq(seq, k=4, stride=1):
    """ Tokenizes the RNA sequence with k-mers.

    Args:
        seg: RNA sequence to be tokenized.
        k: length of the token.
        stride: step when moving the k window on sequence.

    Returns:
        tokens: tokenized sequence.
    """
    seq_length = len(seq)
    tokens = ""

    while seq_length > k:
        tokens += seq[-seq_length:-seq_length+k] + " "
        seq_length -= stride
    tokens += seq[-k:]

    return tokens


def seq_to_tensor(seq, k=4, stride=1):
    """ Converts sequence to tensor.

    The original sequence is nucleotides whose bases are A, T, C and G. This
    function converts it to PyTorch 1-dim tensor(dtype=torch.long) with k-pts
    and stride stride.

    Example:
        >>> seq_to_tensor('ATCGATCG', k=4, stride=1)
        tensor([ 54, 216,  99, 141,  54])

    Args:
        seq: nucleotides sequence, which is supposed to be a string.
        k: length of nucleotide combination. E.g. k of 'ATCGAT' should be 6.
        stride: step when moving the k window on sequence.

    Returns:
        tensor: Pytorch 1-dim tensor representing the sequence.

    """
    seq_length = len(seq)
    tensor = []

    while seq_length > k:
        tensor.append(pattern_to_number(seq[-seq_length:-seq_length+k]))
        seq_length -= stride
    tensor.append(pattern_to_number(seq[-k:]))
    # tensor = torch.IntTensor(tensor)
    tensor = torch.tensor(tensor, dtype=torch.long)

    return tensor


def split_seq(seq, k, stride):
    """ Split nucleotide sequence.

    The original sequence is nucleotides whose bases are A, T, C and G. This
    function split it with k-pts and stride stride.

    Example:
        >>> split_seq('ATCGATCG', k=4, stride=2)
        ['ATCG', 'CGAT', 'ATCG']

    Args:
        seq: nucleotides sequence, which is supposed to be a string.
        k: length of nucleotide combination. E.g. k of 'ATCGAT' should be 6.
        stride: step when moving the k window on sequence.

    Returns:
        splited_seq: list containing splited seqs.

    """
    seq_length = len(seq)
    splited_seq = []

    while seq_length > k:
        splited_seq.append(seq[-seq_length:-seq_length+k])
        seq_length -= stride
    splited_seq.append(seq[-seq_length:])

    return splited_seq


def class_to_tensor(location):
    tensor = torch.zeros(5, dtype=torch.float)
    """ Transforms class name to tensor.

    There are five classes for lncRNA's subcellular location:
        - Cytoplasm
        - Nucleus
        - Ribosome
        - Cytosol
        - Exosome
    This function transforms them to one-hot representation.

    Example:
        >>> class_to_tensor("Exosome")
        tensor([0., 0., 0., 0., 1.])

    Args:
        location: class name for subcellular location. Valid input are listed
                  above.
    Returns:
        tensor: one-hot representation of every class
    """

    if location == "Cytosol":
        tensor[0] = 1
    elif location == "Nucleus" or location == "Nuclear":
        tensor[1] = 1
    elif location == "Ribosome":
        tensor[2] = 1
    elif location == "Cytoplasm":
        tensor[3] = 1
    elif location == "Exosome":
        tensor[4] = 1
    else:
        pass
    return tensor


def num_to_class(num):
    if num == 0:
        return "Cytosol"
    elif num == 1:
        return "Nucleus"
    elif num == 2:
        return "Ribosome"
    elif num == 3:
        return "Cytoplasm"
    elif num == 4:
        return "Exosome"
    else:
        pass


def num_to_class_lncatlas(location):
    if location == "Cytosol":
        tensor[0] = 1
    elif location == "Nucleus" or location == "Nuclear":
        tensor[1] = 1
    elif location == "Ribosome":
        tensor[2] = 1
    elif location == "Cytoplasm":
        tensor[3] = 1
    elif location == "Exosome":
        tensor[4] = 1
    else:
        pass
    return tensor


def embed_from_pretrained(args):
#     if args.embed == 'word2vec':
#         model = word2vec.load(args.kmer_embed_dir)
    if args.embed == 'glove':
        model = load_glove_model(args.kmer_embed_dir)
        print('Read model\n model size: {:d}'.format(len(model)))
    embed = torch.zeros(4**args.k, args.embed_num)
    for i in range(4**args.k):
        try:
            pattern = number_to_pattern(i, args.k)
            embed[i, :] = torch.Tensor(model[pattern])
        except:
            print("Pattern {:s} not found.".format(pattern))
        # if pattern in model.vocab:
        #     embed[i, :] = torch.Tensor(model[pattern])
        # else:
        #     logging.warning('{} not found in embeddings.'.format(pattern))
    return embed


def pattern_to_number(text):
    length = len(text)
    number = 0
    for i in range(length):
        if text[i] == "A":
            number += 0 * 4**(length - i - 1)
        elif text[i] == "C":
            number += 1 * 4**(length - i - 1)
        elif text[i] == "G":
            number += 2 * 4**(length - i - 1)
        elif text[i] == "T":
            number += 3 * 4**(length - i - 1)
        else:
            pass
    return number


def number_to_pattern(num, base):
    pattern = ""
    for i in range(base):
        div = num // 4**(base - i - 1)
        num = num - 4**(base - i - 1) * div
        if div == 0:
            pattern += "A"
        elif div == 1:
            pattern += "C"
        elif div == 2:
            pattern += "G"
        elif div == 3:
            pattern += "T"
        else:
            pass
    return pattern


def get_kmer_frequency(text, base):
    length = len(text) - base + 1
    features = np.zeros(4 ** base, dtype=np.float)
    for i in range(length):
        features[pattern_to_number(text[i:i+base])] += 1
    features /= length
    return features


def collate_fn(data_list):
    code = [i['code'] for i in data_list]
    label = ([i['CNRCI'] for i in data_list],
             [i['is_coding'] for i in data_list])
    return code, label


def collate_fn_biomart(data_list):
    code = [(i['cds'], i['3_utr'], i['5_utr']) for i in data_list]
    label = ([i['CNRCI'] for i in data_list])
    return code, label


def collate_fn_K562(data_list):
    code = [i['code'] for i in data_list]
    label = [[i['CNRCI'],
              i['RCIin'],
              i['RCImem'],
              i['RCIc'],
              i['RCIno'],
              i['RCInp'],
              i['is_coding']] for i in data_list]
    label = torch.tensor(label)
    return code, label


def collate_fn_transformer(data_list):
    code = [i['code'] for i in data_list]
    label = [i['CNRCI'] for i in data_list]
    label = torch.tensor(label)
    return code, label


def collate_fn_kmer(data_list):
    code = [i['code'] for i in data_list]
    label = [i['label'] for i in data_list]
    label = torch.tensor(label)
    return code, label


def collate_fn_cellline(data_list):
    code = [i['code'] for i in data_list]
    label = [i['Value'] for i in data_list]
    label = torch.tensor(label)
    index = [i['cellline'] for i in data_list]
    return code, label, index


def collate_fn_K562_CNRCI(data_list):
    code = [i['code'] for i in data_list]
    label = [[i['CNRCI'],
              0,
              0,
              0,
              0,
              0,
              0] for i in data_list]
    label = torch.tensor(label)
    return code, label


def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def evaluate_metrics(metrics, class_num):
    total_num = torch.sum(metrics)
    total_accuracy = torch.trace(metrics) / total_num
    output = "{}\nConfusion Metrics:\n{:<15s}".format("-"*80, "Pred\\Label")
    for i in range(class_num):
        output += " {:<10s}".format(num_to_class(i))
    for i in range(class_num):
        output += '\n{:<15s}'.format(num_to_class(i))
        for j in range(class_num):
            output += " {:<10d}".format(int(metrics[i][j].tolist()))
    output += "\n{}\nEvaluation Statistics:\n{:<15s} {:<15s} {:<15s} {:<15s} {:<15s}\n".format(
        "-" * 80,
        "Class",
        "Precision",
        "Recall",
        "F1-score",
        "Support"
    )
    eval = torch.zeros(class_num, 4)
    for i in range(class_num):
        eval[i][0] = metrics[i][i] / torch.sum(metrics[i])
        eval[i][1] = metrics[i][i] / torch.sum(metrics[:, i])
        eval[i][2] = 2 * eval[i][0] * eval[i][1] / (eval[i][0] + eval[i][1])
        eval[i][3] = torch.sum(metrics[:, i])
    eval = eval.tolist()
    for i in range(class_num):
        output += \
            "{:<15s} {:<15.4%} {:<15.4%} {:<15.4%} {:<15d}\n".format(
                num_to_class(i),
                eval[i][0],
                eval[i][1],
                eval[i][2],
                int(eval[i][3])
            )
    output += "{}\nTotal accuracy: {:.4%}".format("-" * 80, total_accuracy)
    return output


def evaluate_metrics_sklearn(y_score, y_true, threshold=0):
    # y_true = y_true.astype(np.int)
    y_true = (y_true > threshold).astype(np.int)
    y_pred = (y_score > threshold).astype(np.int)
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    support = y_true.shape[0]
    # print(report)
    return accuracy, precision, recall, roc_auc, report, support


def evaluate_K562(pred, truth):
    C_N_pred, sub_class_pred = analyze_K562(pred)
    C_N_truth, sub_class_truth = analyze_K562(truth)

    C_N_report_pred = \
        metrics.classification_report(C_N_truth, C_N_pred)
    sub_class_report_truth = \
        metrics.classification_report(sub_class_truth, sub_class_pred)

    C_N_acc_pred = \
        metrics.accuracy_score(C_N_truth, C_N_pred)
    sub_class_acc_pred = \
        metrics.accuracy_score(sub_class_truth, sub_class_pred)

    return C_N_report_pred, sub_class_report_truth, C_N_acc_pred, sub_class_acc_pred


def analyze_K562(vec):
    C_N = np.array(vec[:, 0] > 0, dtype=np.float)

    C = vec[:, 1:3]
    N = vec[:, 3:6]

    C_class = np.argmax(C, axis=1)
    N_class = np.argmax(N, axis=1)

    sub_class = C_N * C_class + (1 - C_N) * (N_class + 2)

    return C_N, sub_class


def evaluate_metrics_sklearn_is_coding(y_score, y_true):
    # y_true = y_true.astype(np.int)
    y_true = (y_true > 0.5).astype(np.int)
    y_pred = (y_score > 0.5).astype(np.int)
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)
    # print(report)
    return accuracy, precision, recall, roc_auc, report


def evaluate_all_cellline(preds, labels, cellline, threshold=0):
    df = pd.DataFrame(
        {
            'preds': preds,
            'labels': labels,
            'cellline': cellline
        }
    )

    result = {}
    values = df['cellline'].unique().tolist()
    for i in values:
        cellline_data = df[df['cellline'] == i]
        cellline_preds = np.array(cellline_data['preds'])
        cellline_labels = np.array(cellline_data['labels'])

        accuracy, precision, recall, roc_auc, report, support = \
            evaluate_metrics_sklearn(cellline_preds, cellline_labels, threshold)

        result[label_to_cellline(i)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'report': report,
            'support': support
        }
    return result


def find_nearest(pattern, model):
    vec = model[pattern]
    print(model.pop(pattern))
    ret = min(model, key=lambda x: np.linalg.norm(model[x] - vec))
    return ret


def find_nearest_vector(vec, model):
    ret = min(model, key=lambda x: np.linalg.norm(model[x] - vec))
    return ret


def draw_curve(para_dict, output_dir):
    sns.set_style("ticks")
    plt.figure(figsize=(8, 4), dpi=300)
    df = pd.DataFrame(data=para_dict)
    sns.lineplot(data=df)
    plt.xlabel('Epoch')
    plt.savefig(output_dir)


class AutoLR(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.epoch = 0
        self.loss = []

    def update_lr(self):
        self.epoch += 1
        if self.epoch < 5:
            return self.learning_rate
        else:
            return self.learning_rate


def complementary_seq(seq):
    com_seq = ""
    for i in seq:
        if i == 'A':
            com_seq += 'T'
        elif i == 'T':
            com_seq += 'A'
        elif i == 'C':
            com_seq += 'G'
        elif i == 'G':
            com_seq += 'C'
    return com_seq


def cellline_to_label(cellline):
    cellline_list = [
        'K562',
        'A549',
        'GM12878',
        'H1.hESC',
        'HeLa.S3',
        'HepG2',
        'HT1080',
        'HUVEC',
        'IMR.90',
        'MCF.7',
        'NCI.H460',
        'NHEK',
        'SK.MEL.5',
        'SK.N.DZ',
        'SK.N.SH'
    ]
    return cellline_list.index(cellline)


def label_to_cellline(label):
    cellline_list = [
        'K562',
        'A549',
        'GM12878',
        'H1.hESC',
        'HeLa.S3',
        'HepG2',
        'HT1080',
        'HUVEC',
        'IMR.90',
        'MCF.7',
        'NCI.H460',
        'NHEK',
        'SK.MEL.5',
        'SK.N.DZ',
        'SK.N.SH'
    ]
    return cellline_list[int(label)]


def report_cellline(result):
    report = "{:<10s}{:<10s}{:<10s}{:<10s}\n".format(
        'Cell line', 'Accuracy', 'AUC ROC', 'Support'
    )
    keys = list(result.keys())
    for i in range(15):
        cellline = label_to_cellline(i)
        if cellline in keys:
            report += '{:<10s}{:<10.2%}{:<10.4f}{:<10d}\n'.format(
                cellline,
                result[cellline]['accuracy'],
                result[cellline]['roc_auc'],
                result[cellline]['support']
            )
    return report


if __name__ == "__main__":
    # print(seq_to_tensor("AAAGAAAC"))
    # print(class_to_tensor("Exosome"))
    # a = torch.eye(5)
    # print(evaluate_metrics(a, 5))

    # y_true = np.array([0., 1., 0., 1.])
    # y_score = np.array([0.66, -9., 3., 7])
    # evaluate_metrics_sklearn(y_true, y_score)

    # record = {
    #     'train_loss': [3., 2., 1., 1.5, 1.2],
    #     'train_acc': [0.5, 0.6, 0.7, 0.8, 0.9],
    #     'test_loss': [4., 2., 1.6, 1.4, 1.2],
    #     'test_acc': [0.55, 0.55, 0.6, 0.7, 0.75]
    # }
    # draw_curve(record, 'test.png')

    a = 'ATCGATCG'
    print(complementary_seq(a))

