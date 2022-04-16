import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from data_loader import loadKGData, SLDataset
from model import Net
from utils import evaluate, print_eval_results

if not os.path.exists('./log'):
    os.makedirs('./log')

log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
logging.basicConfig(filename=f"./log/{log_time}.log", level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("run")

def cal_final_result(result_all):
    for m in ['P', 'R', 'N']:
        for top in [10, 20, 50, 100]:
            tmp = []
            for i in [1, 2, 3, 4, 5]:
                tmp.append(result_all['result' + str(i)]['test'][f'{m}{top}'])
            print(f'{m}{top}:{round(np.mean(tmp), 4)},{round(np.std(tmp), 4)}')
            logger.info(f'{m}{top}:{round(np.mean(tmp), 4)},{round(np.std(tmp), 4)}')

def get_partData(data, ratio):
    new_data = []
    for i in range(len(data)):
        random.shuffle(data[i])
        new_data.append(data[i][:int(len(data[i])*ratio)])
    
    print('Number of the origin data: ', len(data[0]))
    print('Number of the partial data: ', len(new_data[0]))

    return np.array(new_data)

def map_genes():
    sl_df = pd.read_csv('./data/sl_data/sl_pair_processed.csv',header=None)
    set_IDa = set(sl_df[0])
    set_IDb = set(sl_df[1])
    list_all = list(set_IDa | set_IDb)
    id2orig = {}
    orig2id = {}
    for i in range(len(list_all)):
        id2orig[i] = int(list_all[i])
        orig2id[list_all[i]] = int(i)
    return id2orig, orig2id

def run(args):
    train_data = np.load('./data/gene_split_cv2/train.npy', allow_pickle=True) 
    valid_data = np.load('./data/gene_split_cv2/valid.npy', allow_pickle=True)
    test_data = np.load('./data/gene_split_cv2/test.npy', allow_pickle=True)

    if args.train_ratio:
        train_data = get_partData(train_data, args.train_ratio)

    gene_kgemb, gene_id = loadKGData()

    id2orig, orig2id = map_genes()
    all_dataset = SLDataset(None, gene_kgemb, gene_id, args.aug_ratio)

    all_gene_feature = []
    for i in id2orig.keys():
        all_gene_feature.append(all_dataset.getFeat(id2orig[i]))
    all_gene_feature = np.asarray(all_gene_feature)

    all_fold_best_result = {}
    for fold_num in range(5):
        print(f"<<<<<<<<<<<<<[ FOLD {fold_num + 1} ]>>>>>>>>>>>>>>>")
        logger.info(f"<<<<<<<<<<<<<[ FOLD {fold_num + 1} ]>>>>>>>>>>>>>>>")

        logging.info('============= Start Training ... ==============')
        print('============= Start Training ... ==============')

        input_size = all_gene_feature.shape[1]
        model = Net(input_size, args)
        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # every 10 steps, lr=gamma*lr

        train_dataset = SLDataset(train_data[fold_num], gene_kgemb, gene_id, args.aug_ratio)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        best_score = -np.inf
        early_stop_cnt = 0

        total_step = 0
        for epoch in range(args.epochs):
            print(f" epoch {epoch + 1} ", end="")
            logger.info(f"===================== epoch {epoch + 1} =====================")
            tic1 = time.time()

            train_loss = []
            for i, (_, gene1_feat, gene1_feat_aug, _, gene2_feat, gene2_feat_aug) in enumerate(train_loader):
                gene1_feat = gene1_feat.to(args.device)
                gene2_feat = gene2_feat.to(args.device)
                gene1_feat_aug = gene1_feat_aug.to(args.device)
                gene2_feat_aug = gene2_feat_aug.to(args.device)

                # Forward
                model.train()
                output = model([gene1_feat.float(), gene2_feat.float(), gene1_feat_aug.float(), gene2_feat_aug.float()])
                batch_loss = model.get_loss(output)
                train_loss.append(batch_loss)
                total_step += 1

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                model._update_target()

            train_loss = torch.mean(torch.stack(train_loss)).data.cpu().numpy()
            toc1 = time.time()
            print(f"train_loss: {train_loss}")
            logger.info(f"train_loss: {train_loss}")
            logger.info(f"each train epoch used {toc1 - tic1}s")

            scheduler.step()

            if (epoch + 1) % 10 == 0:

                print(f"=========== start evaluation ... ============")
                logger.info(f"=========== start evaluation ... =============")

                ts = time.time()
                model.eval()
                eva_result, score_label_mat = evaluate(model, train_data[fold_num], valid_data[fold_num],
                                                       test_data[fold_num], all_gene_feature, args, epoch, fold_num)

                te = time.time()
                logger.info(f"each evalution used {te - ts}s")
                logger.info(f"<<<<<<<<<<<[ EVALUATE RESULT ]>>>>>>>>>>>>")
                logger.info(eva_result)

                if eva_result['valid']['P100'] > best_score:
                    best_score = eva_result['valid']['P100']
                    best_epoch = epoch + 1
                    valid_result = eva_result['valid']
                    test_result = eva_result['test']

                    print(f"<<<<<<<<<<<[ EVALUATE RESULT ]>>>>>>>>>>>>")
                    print_eval_results(eva_result)

                else:
                    early_stop_cnt += 1
                    if early_stop_cnt == args.early_stop:
                        logger.warning('[ EARLY STOP HAPPEN! ]')
                        break

        print(f'<<<<<<<<<<<[ FINAL RESULT {fold_num+1} ]>>>>>>>>>>>>\n')
        print_eval_results({'valid': valid_result, 'test': test_result})
        print(f'BEST PERFORMENCE APPEAR IN {best_epoch}th EPOCH')
        logger.info(f"<<<<<<<<<<<[ FINAL RESULT {fold_num + 1} ]>>>>>>>>>>>>")
        logger.info({'valid': valid_result, 'test': test_result})
        logger.info(f'BEST PERFORMENCE APPEAR IN {best_epoch + 1}th EPOCH')
        all_fold_best_result[f'result{fold_num + 1}'] = {'valid': valid_result, 'test': test_result}

    print('<<<<<<<<<<<[ ALL RESULT ]>>>>>>>>>>>>')
    print()
    for i in range(5):
        print(f'<<<<<[ RESULT {i + 1} ]>>>>>')
        logger.info(f'<<<<<[ RESULT {i + 1} ]>>>>>')
        logger.info(all_fold_best_result[f'result{i + 1}'])
        print_eval_results(all_fold_best_result[f'result{i + 1}'])

    print('<<<<<<<<<<<[ mean,std ]>>>>>>>>>>>>')
    logger.info('<<<<<<<<<<<[ mean,std ]>>>>>>>>>>>>')
    cal_final_result(all_fold_best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_ratio', type=float, default = 0.1, help='percentage of columns for each gene feature')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--early_stop', type=int, default=4, help='the early stop flag')
    parser.add_argument('--epochs', type=int, default=100, help='the number of max epochs')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--latent_size', type=int, default=256, help='latent size for the encoders and predictor')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.995, help='momentum for target encoder optimization')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Percentage of training data used') 
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='optimizer weight decay') 

    args = parser.parse_args()
    logger.info(args)
    if torch.cuda.is_available():
        args.device = torch.device('cuda:%d' % args.gpu)
    else:
        args.device = torch.device('cpu')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    run(args)
