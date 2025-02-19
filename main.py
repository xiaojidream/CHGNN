import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import random
import shutil
import time
import sys
from torch.utils.data import Dataset

from dataset import Dataset as graph_dataset
from utils import get_threshold_auc, calc_metrics

from models.gcn import GCN

from sklearn.metrics import roc_auc_score

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN', type=str,
                    help='baseline of the model')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--n_epoch', default=10, type=int,
                    help='number of epoch to change')
parser.add_argument('--epoch', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('--optimizer', default='Adam', type=str,
                    help='optimizer (Adam)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--batch_size', default=20, type=int,
                    help='mini-batch size (default: 20)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument("--gpu", type=str, default="0", metavar="N",
                    help="input visible devices for training (default: 0)")
parser.add_argument('--seed', default=2, type=int, help='random seed(default: 1)')

def get_graph_data(data_path, mode='train'):
    df_data = pd.read_excel(data_path)
    prob_feat_path = os.path.join(r'.\data\encoder_data', mode + '_probs.csv')
    prob_feat = pd.read_csv(prob_feat_path)
    ids = df_data.iloc[:, 0]
    filter_ids = []
    prob_feat['id'] = prob_feat['id'].astype('str')
    adjacency = []
    edge_attr = []
    node_feat = []
    labels = df_data.iloc[:, 1]
    for i in range(len(ids)):
        id = ids.loc[i]
        # 第i个病例的数据
        patient_i = prob_feat[prob_feat['id'].isin([str(id)])]
        patient_i_sort = patient_i.sort_values('prob')
        patient_i_choose = patient_i_sort

        graph_i = Graph(patient_i_choose)
        edge_attr_i, adj_matrix_i, node_feat_i = graph_i.ad_relationship()
        adjacency.append(adj_matrix_i)
        edge_attr.append(edge_attr_i)
        node_feat.append(node_feat_i)
        filter_ids.append(id)

    return np.array(filter_ids), adjacency, edge_attr, node_feat, np.array(labels)




class Graph(object):
    def __init__(self, patient_df):
        self.count = patient_df.shape[0]
        self.adj_mat = [[None for i in range(self.count)] for i in range(self.count)]
        self.df = patient_df
        self.node_feat = np.zeros((self.count, 33))
        self.base_path =  r'E:\yyx\RanK\OriginData\data\segdata'
        for i in range(self.count):
            feat_idx = [i for i in range(8, 41)]
            self.node = self.df.iloc[i, feat_idx]
            self.node_feat[i] = self.node



    def ad_relationship(self):
        x = np.zeros(self.count)
        x[:self.df.shape[0]] = self.df.iloc[:, 3]
        x = torch.tensor(x) * 0.75
        y = np.zeros(self.count)
        y[:self.df.shape[0]] = self.df.iloc[:, 5]
        y = torch.tensor(y) * 0.75

        z = np.zeros(self.count)
        z[:self.df.shape[0]] = self.df.iloc[:, 7]
        z = torch.tensor(z) * 5
        points = torch.stack([x, y, z], dim=1)
        self.attr = torch.exp(- cal_euclidean(points, points) / 1800)

        edge_attr = np.array(self.attr)
        edge_attr = attr_norm(edge_attr)

        edge_list = open(os.path.join(self.base_path, id + '_edge.txt')).read().splitlines()
        start_idx = []
        end_idx = []
        for row in edge_list:
            start_idx.append(int(row.split(',')[0]))
            end_idx.append(int(row.split(',')[1]))

        N = max(max(start_idx), max(end_idx)) + 1

        adj_matrix = np.zeros((N, N), dtype=int)

        for s, e in zip(start_idx, end_idx):
            adj_matrix[s, e] = 1

        for s, e in zip(start_idx, end_idx):
            adj_matrix[e, s] = 1


        return edge_attr, adj_matrix, self.node_feat


# 计算两个patch之间的空间欧氏距离
def cal_euclidean(a, b):
    a_2 = a ** 2
    b_2 = b ** 2
    sum_a_2 = torch.sum(a_2, dim=1).unsqueeze(1)  # [m, 1]
    sum_b_2 = torch.sum(b_2, dim=1).unsqueeze(0)  # [1, n]
    bt = b.t()
    return sum_a_2 + sum_b_2 - 2 * a.mm(bt)

def attr_norm(A):
    assert A.shape[0] == A.shape[1]
    D = np.sum(A, axis=0)
    d_inv_sqrt = np.power(D, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return A_norm

# Graph dataset
class Graph_dataset(Dataset):
    def __init__(self, id, adjacency, edge_attr, feature, y):
        self.id = id
        self.y = y
        self.adjacency = adjacency
        self.edge_attr = edge_attr
        self.feature = feature

    def __getitem__(self, index):
        case_id = self.id[index]
        adjacency_i = self.adjacency[index]
        edge_attr_i = self.edge_attr[index]
        feature_i = self.feature[index]
        label = self.y[index]
        return torch.Tensor(adjacency_i), torch.Tensor(edge_attr_i), torch.Tensor(feature_i), label, case_id

    def __len__(self):
        return len(self.id)


def aucrocs(output, target):  # 改准确度的计算方式

    """
    Returns:
    List of AUROCs of all classes.
    """
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    AUROCs=roc_auc_score(target_np[:, 0], output_np[:, 0])
    return AUROCs

def val(epoch, model, dataloader, device, type, criterion):
    model.eval()
    n_batch = len(dataloader.dataset) // args.batch_size
    pbar = tqdm(total=n_batch)

    all_loss = []
    y_true = []
    y_score = []

    for _, (adjacency, edge_attr, feature, target, id) in enumerate(train_loader):
        target = target.type(torch.LongTensor).cuda()
        adjacency = adjacency.type(torch.LongTensor).cuda()
        edge_attr = edge_attr.type(torch.FloatTensor).cuda()
        feature = feature.type(torch.FloatTensor).cuda()
        output = model(feature, adjacency, edge_attr)
        y_true.extend(target.cpu().numpy())
        score = nn.functional.softmax(output, dim=1).cpu().detach().numpy()
        y_score.extend(score[:, 1])

        loss = criterion(output, target)
        all_loss.append(float(loss.data.cpu().numpy()))


        pbar.update(1)

    pbar.close()

    loss_value = np.mean(np.array(all_loss))
    roc_auc, best_threshold, _ = get_threshold_auc(y_true, y_score)
    y_pred = (np.array(y_score) >= 0.5).astype(int)
    acc, sensitivity, specificity, ppv, npv = calc_metrics(y_true, y_pred)
    print(
        'epoch:{}, type={}, AUC={}, loss={}, ACC={}, Sensitivity={}, Specificity={}, PPV={}, NPV={}, cut-off={}'
        .format(epoch, type, roc_auc, loss_value, acc, sensitivity, specificity, ppv, npv, best_threshold))
    return loss_value, acc, roc_auc




if __name__ == '__main__':
    global use_cuda
    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()

    # 超参相关
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    learning_rate = args.lr
    weight_decay = args.weight_decay
    total_epoch = args.epoch
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    seed_torch(args.seed)
    # device相关
    device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    # 模型
    model = GCN()
    model = model.to(device)

    # 损失函数
    weights = torch.FloatTensor([1.0, 4.8]).cuda()
    criterion = nn.CrossEntropyLoss(weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 保存路径
    MODELPATH = os.path.join('./results', '1-23-1')



    # 保存路径
    save_path = os.path.join(MODELPATH, "model")
    log_path = os.path.join(MODELPATH, "log")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    patient_id_train, adjacency_train, edge_attr_train, node_feat_train, patient_y_train = get_graph_data(data_path=r'.\data\train.xlsx', mode='train')

    patient_id_val, adjacency_val, edge_attr_val, node_feat_val, patient_y_val = get_graph_data(data_path=r'.\data\val.xlsx', mode='val')

    train_datasets = Graph_dataset(patient_id_train, adjacency_train, edge_attr_train, node_feat_train, patient_y_train)
    val_datasets = Graph_dataset(patient_id_val, adjacency_val, edge_attr_val, node_feat_val, patient_y_val)
    # 数据集
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=batch_size, shuffle=True)


    print("Start Training...")
    # loss
    train_loss = []
    val_loss = []
    # acc
    train_acc = []
    val_acc = []
    # auc
    train_auc = []
    val_auc = []

    for epoch in range(total_epoch):
        start = time.time()
        model.train()
        n_batch = len(train_datasets) // batch_size
        pbar = tqdm(total=n_batch)
        for _, (adjacency, edge_attr, feature, target, id) in enumerate(train_loader):
            target = target.type(torch.LongTensor).cuda()
            adjacency = adjacency.type(torch.LongTensor).cuda()
            edge_attr = edge_attr.type(torch.FloatTensor).cuda()
            feature = feature.type(torch.FloatTensor).cuda()
            output = model(feature, adjacency, edge_attr)
            loss = criterion(output, target)
            optimizer.zero_grad()  # 清除梯度
            loss.backward()
            optimizer.step()
            pbar.update(1)
        pbar.close()

        # 验证训练集
        train_loss_value, train_acc_value, t_auc = val(epoch, model, train_loader, device, type='train', criterion=criterion)
        train_loss.append(train_loss_value)
        train_acc.append(train_acc_value)
        train_auc.append(t_auc)

        # 验证测试集
        val_loss_value, val_acc_value, v_auc = val(epoch, model, val_loader, device, type='val', criterion=criterion)
        val_acc.append(val_acc_value)
        val_loss.append(val_loss_value)
        val_auc.append(v_auc)
        end = time.time()
        print('Running time:{}s.'.format(round(end - start, 4)))
        # 模型保存的逻辑
        if len(val_loss) == 1:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join(save_path, 'first_model' + '.pt'))
            df_acc = pd.DataFrame([],
                                  columns=['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc'])
            df_acc['train_loss'] = train_loss
            df_acc['val_loss'] = val_loss
            df_acc['train_acc'] = train_acc
            df_acc['val_acc'] = val_acc
            df_acc['train_auc'] = train_auc
            df_acc['val_auc'] = val_auc
            df_acc.to_csv(os.path.join(log_path, 'first_result.csv'))
        else:
            if (val_loss[-1] > np.max(val_loss[:-1])):
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join(save_path, 'best_model' + '.pt'))
                df_acc = pd.DataFrame([],
                                      columns=['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc',
                                               'val_auc'])
                df_acc['train_loss'] = [train_loss[-1]]
                df_acc['val_loss'] = [val_loss[-1]]
                df_acc['train_acc'] = [train_acc[-1]]
                df_acc['val_acc'] = [val_acc[-1]]
                df_acc['train_auc'] = [train_auc[-1]]
                df_acc['val_auc'] = [val_auc[-1]]
                df_acc.to_csv(os.path.join(log_path, 'best_result.csv'))

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), },
                   os.path.join(save_path, 'final_model' + '.pt'))
        df_acc = pd.DataFrame([],
                              columns=['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc'])
        df_acc['train_loss'] = train_loss
        df_acc['val_loss'] = val_loss
        df_acc['train_acc'] = train_acc
        df_acc['val_acc'] = val_acc
        df_acc['train_auc'] = train_auc
        df_acc['val_auc'] = val_auc
        df_acc.to_csv(os.path.join(log_path, 'final_result.csv'))
        torch.cuda.empty_cache()

