# This module defines the class of our dataset.

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import utils


class LncRnaDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data = pd.read_csv(dataset_dir)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        # print(self.data.loc[idx]['Loc'])
        item['Loc'] = utils.class_to_tensor(self.data.loc[idx]['Loc'])
        # item['code'] = utils.seq_to_tensor(
        #     self.data.loc[idx]['code'],
        #     k=self.k,
        #     stride=self.stride)
        # item['code_rev'] = utils.seq_to_tensor(
        #     self.data.loc[idx]['code'][::-1],
        #     k=self.k,
        #     stride=self.stride)
        item['code'] = self.data.loc[idx]['code']
        return item


class LncAtlasDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data = pd.read_csv(dataset_dir)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]
        # item['Loc'] = torch.tensor([entry['Chromatin'],
        #                             entry['Insoluble'],
        #                             entry['Membrane'],
        #                             entry['Necleolus'],
        #                             entry['Nucleoplasm']])
        cytoplasm = entry['Cytoplasm']
        nucleus = entry['Nucleus']
        if cytoplasm == nucleus:
            item['Loc'] = torch.Tensor([0.5, 0.5])
        else:
            item['Loc'] = torch.Tensor([cytoplasm/  (cytoplasm + nucleus),
                                        nucleus / (cytoplasm + nucleus)])
        item['code'] = self.data.loc[idx]['code']
        return item


class LncAtlasDatasetRegression(Dataset):
    def __init__(self, dataset_dir, filter=False):
        self.data = pd.read_csv(dataset_dir)

        # self.data = self.data[self.data['code'].str.len() < 500]
        self.data = self.data[(self.data['code'].str.len() < 1000) & (self.data['code'].str.len() > 200)]
        self.data.reset_index(inplace=True, drop=True)

        print('-' * 80)
        print('Loading dataset: ', dataset_dir)
        print('dataset size: ', self.data.shape[0])

        self.mu = np.mean(self.data['Value'].values)
        self.sigma = np.std(self.data['Value'].values)

        print('average: ', self.mu)
        print('standard bias', self.sigma)

        if filter is True:
            print('Processing the filtering...')
            df1 = self.data[self.data['Value'] < -1]
            df2 = self.data[self.data['Value'] > 1]

            self.data = pd.concat([df1, df2])
            self.data.reset_index(inplace=True, drop=True)
            print('dataset size: ', self.data.shape[0])



        print('Load successfully.')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]
        # item['Loc'] = torch.tensor([entry['Chromatin'],
        #                             entry['Insoluble'],
        #                             entry['Membrane'],
        #                             entry['Necleolus'],
        #                             entry['Nucleoplasm']])

        # # For general regression
        # if entry['num'] != 0:
        #     # item['CNRCI'] = torch.sigmoid(5 * (torch.tensor(entry['CNRCI'] / entry['num'] + 1.0699)))
        #     item['CNRCI'] = torch.sigmoid(5 * (torch.tensor(entry['CNRCI'] / entry['num'])))
        # else:
        #     item['CNRCI'] = torch.tensor(0.5).to(torch.float)

        # For the old dataset
        # item['CNRCI'] = torch.tensor(1.0) if entry['Loc'] == 'Nuclear' else torch.tensor(0.0)

        # For mRNA - lncRNA regression
        # item['CNRCI'] = 1 if entry['transcrpts type'] == 'lncRNA' else 0

        # Fo real regression
        # if float(entry['num']) != 0:
        #     item['CNRCI'] = torch.tensor(float(entry['CNRCI']) / float(entry['num']))
        # else:
        #     item['CNRCI'] = torch.tensor(0).to(torch.float)
        
        # For cellline
        item['CNRCI'] = float(entry['Value'])
        item['is_coding'] = 1.0 if entry['transcripts type'] == 'protein_coding' else 0

        item['code'] = entry['code']
        return item

    
class K562Dataset(Dataset):
    def __init__(self, dataset_dir):
        self.data = pd.read_csv(dataset_dir)
        self.columns = list(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]

        RCIs = ['CNRCI', 'RCIc', 'RCIin', 'RCImem', 'RCIno', 'RCInp']
        for RCI in RCIs:
            if RCI in self.columns:
                item[RCI] = float(entry[RCI])

        # item['CNRCI'] = float(entry['CNRCI'])
        # item['RCIc'] = float(entry['RCIc'])
        # item['RCIin'] = float(entry['RCIin'])
        # item['RCImem'] = float(entry['RCImem'])
        # item['RCIno'] = float(entry['RCIno'])
        # item['RCInp'] = float(entry['RCInp'])

        if 'transcripts type' in self.columns:
            item['is_coding'] = 1.0 if entry['transcripts type'] == 'protein_coding' else 0

        item['code'] = utils.complementary_seq(entry['code'])
        return item


class CelllineDataset(Dataset):
    def __init__(self, dataset_dir, filter=False):
        self.data = pd.read_csv(dataset_dir)

        print('-' * 80)
        print('Loading dataset: ', dataset_dir)
        print('dataset size: ', self.data.shape[0])

        self.mu = np.mean(self.data['Value'].values)
        self.sigma = np.std(self.data['Value'].values)

        print('average: ', self.mu)
        print('standard bias', self.sigma)

        if filter is True:
            print('Processing the filtering...')
            df1 = self.data[self.data['Value'] < -1]
            df2 = self.data[self.data['Value'] > 1]

            self.data = pd.concat([df1, df2])
            self.data.reset_index(inplace=True, drop=True)
            print('dataset size: ', self.data.shape[0])

        self.columns = list(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]

        RCIs = ['Value']
        for RCI in RCIs:
            if RCI in self.columns:
                item[RCI] = float(entry[RCI])

        if 'transcripts type' in self.columns:
            item['is_coding'] = 1.0 if entry['transcripts type'] == 'protein_coding' else 0
        if 'Data Source' in self.columns:
            item['cellline'] = utils.cellline_to_label(entry['Data Source'])
        if 'code' in self.columns:
            item['code'] = entry['code']

        return item



class BiomartDataset(Dataset):
    def __init__(self, dataset_dir, nc=True, filter=False):
        self.data = pd.read_csv(dataset_dir)
        self.data.dropna(subset=['Value'], inplace=True)
        self.data.reset_index(inplace=True)


        if nc is True:
            self.data = self.data[self.data['Biotype'] == 'nc']
            self.data.reset_index(inplace=True)

        print('-' * 80)
        print('Loading dataset: ', dataset_dir)
        print('dataset size: ', self.data.shape[0])

        self.mu = np.mean(self.data['Value'].values)
        self.sigma = np.std(self.data['Value'].values)

        print('average: ', self.mu)
        print('standard bias', self.sigma)

        if filter is True:
            print('Processing the filtering...')
            df1 = self.data[self.data['Value'] < -1]
            df2 = self.data[self.data['Value'] > 1]

            self.data = pd.concat([df1, df2])
            self.data.reset_index(inplace=True)
            print('dataset size: ', self.data.shape[0])

        self.columns = list(self.data)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]

        item['CNRCI'] = entry['Value']

        item['cds'] = entry['cds']
        item['3_utr'] = entry['3_utr']
        item['5_utr'] = entry['5_utr']

        return item



class kmer_dataset(Dataset):
    def __init__(self, dataset_dir, filter=False):
        self.data = pd.read_csv(dataset_dir)

        if filter is True:
            self.data = self.data[self.data['code'].str.len() < 20000]
            self.data.reset_index(inplace=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = {}
        entry = self.data.loc[idx]

        item['code'] = entry['code']
        item['label'] = utils.get_kmer_frequency(entry['code'], 4)

        return item

if __name__ == "__main__":
    data = LncRnaDataset("./data_preprocess/lncRNA_dataset.csv")
    print(data[0])
