# NSF4SL: Negative-Sample-Free Contrastive Learning for Ranking Synthetic Lethal Partner Genes in Human Cancers
- This is the code for our paper ``NSF4SL: Negative-Sample-Free Contrastive Learning for Ranking Synthetic Lethal Partner Genes in Human Cancers''.
- NSF4SL is a negative-sample-free model for prediction of synthetic lethality (SL) based on a self-supervised contrastive learning framework. 

## Overview
 ![NSF4SL pipeline](https://github.com/JieZheng-ShanghaiTech/NSF4SL/raw/main/figs/pipeline.png)
 
## Install
```
git clone git@github.com:JieZheng-ShanghaiTech/NSF4SL.git
cd NSF4SL
pip install -r requirements.txt
```
## Data
**SL Data**

We test the model performance under 3 kinds of 5-fold cross-validation:
- CV1: we split the train/test data by SL pairs, and both genes of a testing pair would be likely to appear
in the training set. 
  - data path: .data/pair_split_cv1
- CV2: the data is split by genes, where only one gene of a testing pair is present in the training set.
  - data path: .data/gene_split_cv2
- CV3: the data is split by genes, where neither genes of a testing pair is present in the training set. 
  - data path: .data/gene_split_cv3
 
The format of one input sample is `(gene1, gene2)`. Please modify the data path in main.py if you would like to test different CVs.

**Knowledge Graph Embedding**

We train the SynLethKG embedding based on [DGL-KE](https://dglke.dgl.ai/doc/index.html). The format of one input sample is `(entity1, relation, entity2)`. The pre-trained KG embedding is under the data path: .data/kg_embed.

## Example
```
python main.py 
    --aug_ratio=0.1         # ratio of augmentraion for each gene feature
    --batch_size=256        # batch size for training
    --gpu=0                 # ID of GPU
    --latent_size=256       # latent size for encoders and predictor
    --train_ratio=None      # ratio of training data
```


## Acknowledgement
The code is inspired by [BUIR](https://github.com/donalee/BUIR).
>[Bootstrapping user and item representations for one-class collaborative filtering](https://arxiv.org/pdf/2105.06323.pdf) <br />
Lee, Dongha, et al. "Bootstrapping user and item representations for one-class collaborative filtering." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021.

SL data and SynLethKG are constructed based on [SynLethDB 2.0](https://www.biorxiv.org/content/10.1101/2021.12.28.474346v1.full.pdf).
>[SynLethDB 2.0: A web-based knowledge graph database on synthetic lethality for novel anticancer drug discovery](https://www.biorxiv.org/content/10.1101/2021.12.28.474346v1.full.pdf) <br />
Wang, Jie, et al. "SynLethDB 2.0: A web-based knowledge graph database on synthetic lethality for novel anticancer drug discovery." bioRxiv (2021).
