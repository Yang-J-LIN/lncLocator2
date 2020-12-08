import argparse


def parse_args():
    """ Sets the arguments needed for the model.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset',
                        default="CNRCI/CNRCI_train_data_source/transcripts_type/H1.hESC/lncRNA.csv")
    parser.add_argument('--dev_dataset',
                        default="CNRCI/CNRCI_dev_data_source/transcripts_type/H1.hESC/lncRNA.csv")
    parser.add_argument('--test_dataset',
                         default="CNRCI/CNRCI_test_data_source/transcripts_type/H1.hESC/lncRNA.csv")
    parser.add_argument('--kmer_dataset',
                         default="dataset/raw_data/gencode.v32.transcripts.filtered.sorted_by_ENSG.csv")

    parser.add_argument('--kmer_embed_dir',
                        default='glove/gencode.v32.transcripts.6.3.16.glove.txt')
    parser.add_argument('--embed',
                        default='glove')
    parser.add_argument('--k',
                        default=6,
                        type=int)
    parser.add_argument('--save',
                        default='checkpoints/')
    parser.add_argument('--stride',
                        default=3,
                        type=int)

    # model arguments
    parser.add_argument('--hidden_dim',
                        default=32,
                        type=int)
    parser.add_argument('--class_num',
                        default=2,
                        type=int)
    parser.add_argument('--embed_num',
                        default=16,
                        type=int)
    parser.add_argument('--onehot',
                        default=False,
                        type=bool)
    parser.add_argument('--model',
                        default='cnn_bilstm')
    parser.add_argument('--cnn_out_channels',
                        default=16,
                        type=int)
    parser.add_argument('--cnn_stride',
                        default=2,
                        type=int)
    parser.add_argument('--kernel_size',
                        default=12,
                        type=int)
    # training arguments
    parser.add_argument('--mode',
                        default='regression')
    parser.add_argument('--epochs',
                        default=100,
                        type=int)
    parser.add_argument('--batchsize',
                        default=32,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float)
    parser.add_argument('--weight_decay',
                        default=1e-2,
                        type=float)
    parser.add_argument('--optimizer',
                        default='adam')
    parser.add_argument('--dropout',
                        default=0.5,
                        type=float)
    parser.add_argument('--dropout_LC',
                        default=0.3,
                        type=float)
    parser.add_argument('--device',
                        default='cuda:1')
    parser.add_argument('--freeze_embed',
                        default=True,
                        type=bool)                       
    # testing arguments
    parser.add_argument('--eval_dataset',
                        default="dataset/cd-hit/dataset_expanded_3/dataset/\
lncRNA_dataset_test.csv")
    parser.add_argument('--model_path',
                        default='checkpoints/model.pth')
    parser.add_argument('--test_batchsize',
                        default=256,
                        type=int)
    args = parser.parse_known_args()[0]
    return args
