import copy
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random


class SLDataset(Dataset):
    def __init__(self, sl_pair, kgemb_data, gene_id, aug_ratio):
        super(SLDataset, self).__init__()
        self.sl_pair = sl_pair
        self.aug_ratio = aug_ratio
        self.kgemb_data, self.gene_id =  kgemb_data, gene_id
        self.kgemb_data_mean = np.mean(self.kgemb_data, axis=0)

        geneid2index = {}
        geneid2index_kgemb = {} # transE embedding
        for i in range(len(self.gene_id)):
            geneid2index[self.gene_id[i]] = i
        self.geneid2index = geneid2index

        kgemb = pd.read_csv('./data/kg_embed/entities.tsv', sep='\t', header=None)
        for idx, row in kgemb.iterrows():
            geneid2index_kgemb[row[1]] = row[0]
        self.geneid2index_kgemb = geneid2index_kgemb

    def __len__(self):
        return len(self.sl_pair)

    def __getitem__(self, index):
        gene1_id = int(self.sl_pair[index][0])
        gene2_id = int(self.sl_pair[index][1])
        gene1_feat = self.getFeat(gene1_id)
        gene2_feat = self.getFeat(gene2_id)

        # AUG
        id_lst = list(range(gene1_feat.shape[0]))
        random.shuffle(id_lst)
        gene1_maskid = id_lst[:int(len(id_lst)*self.aug_ratio)]
        random.shuffle(id_lst)
        gene2_maskid = id_lst[:int(len(id_lst)*self.aug_ratio)]
        gene1_feat_aug = copy.deepcopy(gene1_feat)
        gene2_feat_aug = copy.deepcopy(gene2_feat)
        
        for id in gene1_maskid:
            gene1_feat_aug[id] = self.kgemb_data_mean[id]
        for id in gene2_maskid:
            gene2_feat_aug[id] = self.kgemb_data_mean[id]

        return gene1_id, gene1_feat, gene1_feat_aug, gene2_id, gene2_feat, gene2_feat_aug

    def getFeat(self, gene_id):
        index = self.geneid2index[gene_id]
        gene_kg = self.kgemb_data[self.geneid2index_kgemb[gene_id]]
        gene_feature = gene_kg

        return gene_feature


def loadKGData():
    print('Loading TransE data...')
    gene_id = np.load('./data/sl_data/gene_id.npy')
    kgemb_data = np.load('./data/kg_embed/kg_TransE_l2_entity.npy') # kg id

    return kgemb_data, gene_id