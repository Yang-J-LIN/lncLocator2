# This module provides train(), develop() and test()
import logging

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from scipy import stats

import utils

from torch.utils.data.sampler import WeightedRandomSampler

logging.basicConfig(filename="train.log",
                    filemode="a",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.INFO)


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
        logging.info("Trainer informations:")
        logging.info("Model information:\n{}".format(model))
        logging.info("Optimizer information:\n{}".format(optimizer))
        logging.info("Criterion information:\n{}".format(criterion))
        logging.info("Device information:\n{}".format(device))
        model_paras = ""
        paras = model.named_parameters()
        for name, para in paras:
            if para.requires_grad:
                model_paras += "{}:{}\n".format(name, para.size())
        logging.info("Parameters information:\n{}".format(model_paras))

    def train(self, dataset):
        logging.info("Start training...")
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        print("LEN", len(dataset))
        # You can adjust the weight according to the CNRCI value
        weights = [0.5 if dataset[i]['CNRCI'] > 0.5 else 0.5
                   for i in range(len(dataset))]
        sampler = WeightedRandomSampler(weights,
                                        num_samples=len(dataset),
                                        replacement=True)

        dataloader = DataLoader(dataset,
                                self.args.batchsize,
                                collate_fn=utils.collate_fn,
                                sampler=sampler)

        num = 0
        for idx, (code, label) in enumerate(dataloader):

            CNRCI_label = torch.tensor(label[0]).to(self.args.device)
            is_coding_label = torch.tensor(label[1]).to(self.args.device).to(torch.float)

            self.optimizer.zero_grad()

            input_code = [utils.seq_to_tensor(
                i,
                k=self.args.k,
                stride=self.args.stride) for i in code]

            CNRCI_pred, is_coding_pred = self.model(input_code)
            
            classification_loss_fn = nn.BCEWithLogitsLoss()

            class_label = (CNRCI_label > 0).to(torch.float32)

            max_weight = stats.norm(dataset.mu, dataset.sigma).pdf(dataset.mu)
            weight = stats.norm(dataset.mu, dataset.sigma).pdf(CNRCI_label.cpu().numpy())

            # The implementation may be different from the paper
            weight = torch.tensor(((max_weight - weight) / max_weight),
                                  device=self.args.device) + 0.5
            
            # weight_sign = (((CNRCI_pred.detach() * CNRCI_label) < 0) + 3).to(torch.float32) / 4

            if self.args.mode == 'classification':
                loss = classification_loss_fn(CNRCI_pred, class_label)
            else:
                # loss = torch.mean(((CNRCI_pred - CNRCI_label) ** 2) * weight * weight_sign)
                loss = torch.mean(((CNRCI_pred - CNRCI_label) ** 2) * weight)

            # Output some information regularly
            if idx % 50 == 0:
                print(
                    "%d/%d" % (idx * self.args.batchsize, len(dataset)),
                    "\nloss:", loss
                )
                print(CNRCI_label)
                print(CNRCI_pred)

            total_loss += loss.item()
            loss.backward()

            tmp = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()

            num = idx
        self.epoch += 1
        return total_loss / num

    def test(self, dataset):

        self.model.eval()

        batchsize = self.args.test_batchsize

        with torch.no_grad():
            total_loss = 0
            num = 0

            # Initialize list to record CNRCI's preds and labels
            CNRCI_predictions = \
                torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            CNRCI_ground_truths = \
                torch.zeros(len(dataset), dtype=torch.float, device='cpu')

            # Initialize list to record is_coding's preds and labels
            is_coding_predictions = \
                torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            is_coding_ground_truths = \
                torch.zeros(len(dataset), dtype=torch.float, device='cpu')

            # Initialize Dataloader to randomly sample data from dataset
            dataloader = DataLoader(dataset,
                                    batchsize,
                                    shuffle=False,
                                    collate_fn=utils.collate_fn)

            # Initialize the loss criterions for RNA subcellular localization
            # and ncRNA identification
            CNRCI_criterion = nn.MSELoss()
            is_coding_criterion = nn.BCELoss()

            classification_loss_fn = nn.BCEWithLogitsLoss()

            # Start test
            for idx, (code, label) in enumerate(dataloader):

                # Move the labels to gpu
                CNRCI_label = torch.tensor(label[0]).to(self.args.device)
                is_coding_label = torch.tensor(label[1]).to(self.args.device)

                # Record the labels
                assert CNRCI_label.shape[0] == is_coding_label.shape[0]
                batchsize_real = CNRCI_label.shape[0]

                CNRCI_ground_truths[idx*batchsize:
                                    idx*batchsize+batchsize_real] = \
                    CNRCI_label.to('cpu').squeeze()

                is_coding_ground_truths[idx*batchsize:
                                        idx*batchsize+batchsize_real] = \
                    is_coding_label.to('cpu').squeeze()

                num = idx
                print("Evaluating epoch {}, {}/{}".format(self.epoch,
                                                          idx*batchsize,
                                                          len(dataset)),
                      end='\r',
                      flush=True)

                # Splice RNA seq with k-mer and stride
                input_code = [utils.seq_to_tensor(
                    i,
                    k=self.args.k,
                    stride=self.args.stride) for i in code]

                class_label = (CNRCI_label > 0).to(torch.float32)

                # Get the predictions from model
                CNRCI_pred, is_coding_pred = self.model(input_code)

                if self.args.mode == 'classification':
                    loss = classification_loss_fn(CNRCI_pred, class_label)
                else:
                    loss = self.criterion(CNRCI_pred.squeeze(), CNRCI_label)

                # Record the total loss
                total_loss += loss.item() * CNRCI_pred.shape[0]

                # Record the predictions
                CNRCI_predictions[idx*batchsize:
                                  idx*batchsize+batchsize_real] = \
                    CNRCI_pred.to('cpu').squeeze()
                is_coding_predictions[idx*batchsize:
                                      idx*batchsize+batchsize_real] = \
                    is_coding_pred.to('cpu').squeeze()

            predictions = (CNRCI_predictions, is_coding_predictions)
            ground_truths = (CNRCI_ground_truths, is_coding_ground_truths)

        return total_loss / len(dataset), predictions, ground_truths

    def evaluate(self, seq, label):
        writer = SummaryWriter(log_dir='./log/log_test', comment=self.args.model)
        input_code = [utils.seq_to_tensor(
                      seq,
                      k=self.args.k,
                      stride=self.args.stride)]
        self.model.eval()
        CNRCI_pred, is_coding_pred = self.model(input_code)
        print('Pred:', CNRCI_pred)
        print('Label:', label)
        writer.add_graph(self.model, (input_code,), verbose=False)
        writer.add_scalar('aaa', 1, 0)
        writer.close()
