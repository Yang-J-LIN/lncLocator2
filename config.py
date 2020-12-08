import argparse


def parse_args():
    """ Sets the arguments needed for the model.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        default='./test_input.fa')
    parser.add_argument('--kmer_embed_dir',
                        default='./glove/gencode.v32.transcripts.6.3.16.glove.txt')
    # default='dataset/lncRNA_dataset_6_3/1568620703.fas_expanded_6_3.bin')
    parser.add_argument('--embed',
                        default='glove')
    parser.add_argument('--model_dir',
                        default='./checkpoints/')
    parser.add_argument('--k',
                        default=6,
                        type=int)
    parser.add_argument('--stride',
                        default=3,
                        type=int)
    parser.add_argument('--cellline',
                        default='H1.hESC')
    parser.add_argument('--timeVar',
                        default=None)
    parser.add_argument('--email',
                        default=None)
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
                        # default='transformer')
    parser.add_argument('--cnn_out_channels',
                        default=16,
                        type=int)
    parser.add_argument('--cnn_stride',
                        default=2,
                        type=int)
    parser.add_argument('--kernel_size',
                        default=12,
                        type=int)
    parser.add_argument('--dropout',
                        default=0,
                        type=float)
    parser.add_argument('--dropout_LC',
                        default=0,
                        type=float)
    # training arguments
    parser.add_argument('--mode',
                        default='regression')
    parser.add_argument('--device',
                        default='cpu')
    parser.add_argument('--freeze_embed',
                        default=True,
                        type=bool)                       
    args = parser.parse_known_args()[0]
    return args
