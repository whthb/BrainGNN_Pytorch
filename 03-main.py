import os
import numpy as np
import argparse
import time
import copy

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from imports.ABIDEDataset import ABIDEDataset
from torch_geometric.loader import DataLoader
from net.braingnn import Network
from imports.utils import train_val_test_split
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=100, help='size of the batches')
parser.add_argument('--dataroot', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/ABIDE_pcp/cpac/filt_noglobal'), help='root directory of the dataset')
parser.add_argument('--processed_file', type=str, default='data_pyg2.pt', help='PyG processed cache filename')
parser.add_argument('--edge_source', type=str, default='pcorr', choices=['pcorr', 'corr'], help='connectivity matrix used to build graph edges')
parser.add_argument('--edge_topk', type=int, default=None, help='keep top-k absolute edges per ROI before symmetrizing; default keeps all edges')
parser.add_argument('--edge_top_percent', type=float, default=None, help='keep this fraction of strongest edges per ROI before symmetrizing')
parser.add_argument('--positive_edges_only', action='store_true', help='keep only positive connectivity values as edge weights')
parser.add_argument('--best_metric', type=str, default='loss', choices=['loss', 'acc', 'balanced_acc'], help='validation metric used to save the best checkpoint')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=200, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--dim1', type=int, default=32, help='first GNN hidden dimension')
parser.add_argument('--dim2', type=int, default=32, help='second GNN hidden dimension')
parser.add_argument('--fc_dim', type=int, default=512, help='fully connected hidden dimension')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
parser.add_argument('--log_path', type=str, default='./log/', help='path to save TensorBoard logs')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
name = 'ABIDE'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join(opt.log_path,str(fold)))



################## Define Dataloader ##################################

dataset = ABIDEDataset(path, name, processed_filename=opt.processed_file,
                       edge_source=opt.edge_source, edge_topk=opt.edge_topk,
                       edge_top_percent=opt.edge_top_percent,
                       positive_edges_only=opt.positive_edges_only)
dataset_data = dataset._data if hasattr(dataset, '_data') else dataset.data
dataset_data.y = dataset_data.y.squeeze()
dataset_data.x = torch.nan_to_num(dataset_data.x, nan=0.0, posinf=0.0, neginf=0.0)

tr_index,val_index,te_index = train_val_test_split(fold=fold, labels=dataset_data.y.cpu().numpy())
train_dataset = dataset[tr_index.tolist()]
val_dataset = dataset[val_index.tolist()]
test_dataset = dataset[te_index.tolist()]


train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)



############### Define Graph Deep Learning Network ##########################
model = Network(opt.indim,opt.ratio,opt.nclass, dim1=opt.dim1,
                dim2=opt.dim2, fc_dim=opt.fc_dim,
                dropout=opt.dropout).to(device)
print(model)

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
        writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
    return loss_all / len(train_dataset), s1_arr, s2_arr ,w1,w2


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    preds = []
    trues = []
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        preds.append(pred.cpu().detach().numpy())
        trues.append(data.y.cpu().detach().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return correct / len(loader.dataset), balanced_accuracy_score(trues, preds)

def test_loss(loader,epoch,stage='testing'):
    print(stage + '...........')
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_score = -1e10
for epoch in range(0, num_epoch):
    since  = time.time()
    tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
    tr_acc, tr_bacc = test_acc(train_loader)
    val_acc, val_bacc = test_acc(val_loader)
    val_loss = test_loss(val_loader,epoch,stage='validating')
    time_elapsed = time.time() - since
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Train Balanced Acc: {:.7f}, '
          'Val Loss: {:.7f}, Val Acc: {:.7f}, Val Balanced Acc: {:.7f}'.format(
              epoch, tr_loss, tr_acc, tr_bacc, val_loss, val_acc, val_bacc))

    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc,
                              'train_bacc':tr_bacc,'val_bacc':val_bacc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
    writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
    writer.add_histogram('Hist/hist_s2', s2_arr, epoch)

    if opt.best_metric == 'loss':
        current_score = -val_loss
    elif opt.best_metric == 'acc':
        current_score = val_acc
    else:
        current_score = val_bacc

    if current_score > best_score and epoch > 5:
        print("saving best model")
        best_score = current_score
        best_model_wts = copy.deepcopy(model.state_dict())
        if save_model:
            torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))
    scheduler.step()

#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################

if opt.load_model:
    model = Network(opt.indim,opt.ratio,opt.nclass, dim1=opt.dim1,
                    dim2=opt.dim2, fc_dim=opt.fc_dim,
                    dropout=opt.dropout).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth'),
                                     map_location=device, weights_only=True))
    model.eval()
    preds = []
    correct = 0
    for data in val_loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(1)[1]
        preds.append(pred.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
    preds = np.concatenate(preds,axis=0)
    trues = val_dataset.data.y.cpu().detach().numpy()
    cm = confusion_matrix(trues,preds)
    print("Confusion matrix")
    print(classification_report(trues, preds))

else:
   model.load_state_dict(best_model_wts)
   model.eval()
   test_accuracy, test_bacc = test_acc(test_loader)
   test_l= test_loss(test_loader,0)
   print("===========================")
   print("Test Acc: {:.7f}, Test Balanced Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_bacc, test_l))
   print(opt)
