import os
import shutil
import logging

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from torch.optim.lr_scheduler import ReduceLROnPlateau
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import config

from model.cnn_bilstm.cnn_bilstm_2 import CNN_BiLSTM, Bitask_learning

import dataset
import train

logging.basicConfig(filename="train.log",
                    filemode="a",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.INFO)


def main():
    args = config.parse_args()

    device = torch.device(args.device)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.onehot is True:
        args.embed_num = 4
        args.kernel_size = 4
        args.k = 1
        args.stride = 1

    model_path = os.path.join(args.save, 'model.pth')

    # Prepare the dataset
    # Train set
    train_dataset = dataset.LncAtlasDatasetRegression(args.train_dataset, True)
    logging.info(
        "\nLoad dataset: {:s}\nDataset size: {:d}".format(
            args.train_dataset, len(train_dataset)
        )
    )
    # Development set
    dev_dataset = dataset.LncAtlasDatasetRegression(args.dev_dataset, True)
    logging.info(
        "\nLoad dataset: {:s}\nDataset size: {:d}".format(
            args.dev_dataset, len(dev_dataset)
        )
    )
    # Test set
    test_dataset = dataset.LncAtlasDatasetRegression(args.test_dataset, True)
    logging.info(
        "\nLoad dataset: {:s}\nDataset size: {:d}".format(
            args.test_dataset, len(test_dataset)
        )
    )

    # Prepare the Training
    # Select model
    model = Bitask_learning(args)
    model.cuda(device)

    # Select loss function
    criterion = nn.MSELoss()

    # Select optimizer 
    if args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sparseadam':
        optimizer = optim.SparseAdam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

    # Make training strategy -- reduce learning rate
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=1, patience=20)


    trainer = train.Trainer(args, model, criterion, optimizer, device)

    min_loss = float('inf')
    max_auroc = 0

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.initialize()

    loss_record = {
        'train_loss': [],
        'dev_loss': [],
        'test_loss': []
    }

    acc_record = {
        'train_acc': [],
        'dev_acc': [],
        'test_acc': []
    }

    auroc_record = {
        'train_auroc': [],
        'dev_auroc': [],
        'test_auroc': []
    }


    # Training process
    for epoch in range(args.epochs):

        if os.path.exists(model_path + '.tmp') is False:
            _ = trainer.train(train_dataset)

            torch.save(model.state_dict(), model_path + '.tmp')

        logging.info('Evaluation after epoch {}'.format(epoch + 1))

        train_loss, train_preds, train_labels = trainer.test(train_dataset)

        train_preds_CNRCI, _ = train_preds
        train_labels_CNRCI, _ = train_labels

        train_accuracy, train_precision, train_recall, train_roc_auc, train_report, train_support = \
            utils.evaluate_metrics_sklearn(train_preds_CNRCI.numpy(), train_labels_CNRCI.numpy())
        logging.info(
            "{:s}Train: Classfication report:\n{:s}\nROC AUC score:{:.5g}{:s}".format(
                '\n' + '-' * 80 + '\n',
                train_report,
                train_roc_auc,
                '\n' + '-' * 80
            ))


        dev_loss, dev_preds, dev_labels = trainer.test(dev_dataset)

        dev_preds_CNRCI, _ = dev_preds
        dev_labels_CNRCI, _ = dev_labels

        dev_accuracy, dev_precision, dev_recall, dev_roc_auc, dev_report, dev_support = \
            utils.evaluate_metrics_sklearn(dev_preds_CNRCI.numpy(), dev_labels_CNRCI.numpy())
        logging.info(
            "{:s}dev: CNRCI classfication report:\n{:s}\nROC AUC score:{:.5g}{:s}".format(
                '\n' + '-' * 80 + '\n',
                dev_report,
                dev_roc_auc,
                '\n' + '-' * 80
            ))


        test_loss, test_preds, test_labels = trainer.test(test_dataset)

        test_preds_CNRCI, _ = test_preds
        test_labels_CNRCI, _ = test_labels

        test_accuracy, test_precision, test_recall, test_roc_auc, test_report, test_support = \
            utils.evaluate_metrics_sklearn(test_preds_CNRCI.numpy(), test_labels_CNRCI.numpy())
        logging.info(
            "{:s}test: CNRCI classfication report:\n{:s}\nROC AUC score:{:.5g}{:s}".format(
                '\n' + '-' * 80 + '\n',
                test_report,
                test_roc_auc,
                '\n' + '-' * 80
            ))

        loss_record['train_loss'].append(train_loss)
        acc_record['train_acc'].append(train_accuracy)
        auroc_record['train_auroc'].append(train_roc_auc)

        loss_record['dev_loss'].append(dev_loss)
        acc_record['dev_acc'].append(dev_accuracy)
        auroc_record['dev_auroc'].append(dev_roc_auc)
        
        loss_record['test_loss'].append(test_loss)
        acc_record['test_acc'].append(test_accuracy)
        auroc_record['test_auroc'].append(test_roc_auc)

        record = dict(loss_record,
                      **acc_record,
                      **auroc_record)
        
        record = pd.DataFrame(record)
        record.to_csv('record.csv')
        

        utils.draw_curve(loss_record, 'loss.png')
        utils.draw_curve(acc_record, 'acc.png')
        utils.draw_curve(auroc_record, 'auroc.png')

        scheduler.step(train_accuracy)

        print('train loss:', train_loss,
              'dev loss:', dev_loss)


        logging.info(
            'train loss:{:.5g}\tdev loss:{:.5g}'.format(
                train_loss, dev_loss
            ))

        if os.path.exists(model_path):
            if test_roc_auc > max_auroc:
                shutil.copy(model_path, model_path + '.bak')
                max_auroc = test_roc_auc

        if dev_loss < min_loss:
            min_loss = dev_loss

            shutil.move(model_path + '.tmp', model_path)
        else:
            os.remove(model_path + '.tmp')


if __name__ == "__main__":
    main()
