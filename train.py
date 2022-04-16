'''
Ref: https://github.com/hwwang55/KGNN-LS/blob/master/src/train.py
for i, (label, inp) in enumerate(loader):
    = model(adj_entity, adj_relation, inp)
'''
import torch
import copy
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc

def train(model, optimizer, train_loader, dev_loader, args):
    m = torch.nn.Sigmoid()
    criterion = torch.nn.BCELoss()
    max_auc = 0
    for epoch in range(args.epochs):
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # forward
            output = model(inp)
            log = torch.squeeze(m(output))
            # define loss
            loss1 = criterion(label, log)
            loss_train = loss1
            # backward
            loss_train.backward()
            # optimizer
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            label_batch = label_ids.flatten().tolist()
            output_batch = output.flatten().tolist()

            output_batch[output_batch >= 0.5] = 1
            output_batch[output_batch < 0.5] = 0

            y_label_train = y_label_train + label_batch
            y_pred_train = y_pred_train + output_batch



            # metrics
            auc_batch = roc_auc_score(label_batch, output_batch)
            p, r, t = precision_recall_curve(label_batch, output_batch)
            aupr_batch = auc(r, p)
            train_f1 = f1_score(label_batch, output_batch)


            if i % 1000 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))
        # 
        roc_train = roc_auc_score(y_label_train, y_pred_train)

        # validation
        roc_val, prc_val, f1_val, loss_val = test(model, dev_loader, args)
        if roc_val > max_auc:
            model_max = copy.deepcopy(model)
            max_auc = roc_val

        print('epoch: {:04d}'.format(epoch + 1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'auroc_train: {:.4f}'.format(roc_train),
                'loss_val: {:.4f}'.format(loss_val.item()),
                'auroc_val: {:.4f}'.format(roc_val),
                'auprc_val: {:.4f}'.format(prc_val),
                'f1_val: {:.4f}'.format(f1_val))


        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


def test(model, loader, args):
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    # lbl = data_a.y

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            if args.cuda:
                label = label.cuda()

            output, _ = model(inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.float())

            loss = loss1

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss
