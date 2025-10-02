import torch
import argparse
import os
from dataloader import *
from network import *
from Nmetrics import evaluate
from tqdm import tqdm
import sys
import signal
import pandas as pd
from sklearn.cluster import KMeans
import warnings
import time
import yaml
warnings.filterwarnings("ignore")


def seed_everything(SEED=42):  # Set random seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

class Logger(object):
    def __init__(self, file_name="log_output.txt", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

def signal_handler(sig, frame):
    print("\nProgram interrupted, saving logs...")
    logger.log.close()
    sys.exit(0)

# Setup signal handler
signal.signal(signal.SIGINT, signal_handler)

def pd_toExcel(my_dic, fileName):  # Save data to an Excel file using pandas
    Mrs, Eps, Class, ACCs, NMIs, PURs, Fscores, Precisions, Recalls, ARIs = [], [], [], [], [], [], [], [], [], []
    for i in range(len(my_dic)):
        Mrs.append(my_dic[i]["Missrate"])
        Eps.append(my_dic[i]["Epoch"])
        Class.append(my_dic[i]["Result_class"])
        ACCs.append(my_dic[i]["ACC"])
        NMIs.append(my_dic[i]["NMI"])
        PURs.append(my_dic[i]["PUR"])
        Fscores.append(my_dic[i]["Fscore"])
        Precisions.append(my_dic[i]["Prec"])
        Recalls.append(my_dic[i]["Recall"])
        ARIs.append(my_dic[i]["ARI"])

    dfData = {  # A dictionary for the DataFrame
        'Missrate': Mrs,
        'Epoch': Eps,
        'Result_class': Class,
        'ACC': ACCs,
        'NMI': NMIs,
        'Purity': PURs,
        'Fscore': Fscores,
        'Prec': Precisions,
        'Recall': Recalls,
        'ARI': ARIs,
    }
    df = pd.DataFrame(dfData)  # Create DataFrame
    df.to_excel(fileName, index=False)  # Save to excel, without index column


def get_SingleHs(cluster_loader):  # Get latent representations (H) for each view
    singleHs = [[] for _ in range(args.view_num)]
    with torch.no_grad():
        for batch_idx, (xs, _, _, _) in enumerate(cluster_loader):
            xs = [item.to(device) for item in xs]
            hs = model.get_Single_reconHs(xs)
            for v in range(args.view_num):
                singleHs[v] += hs[v].cpu().tolist()
    return singleHs

def get_SingleZs(cluster_loader):  # Get contrastive representations (Z) for each view
    singleZs = [[] for _ in range(args.view_num)]
    with torch.no_grad():
        for batch_idx, (xs, _, _, _) in enumerate(cluster_loader):
            xs = [item.to(device) for item in xs]
            zs = model.get_Single_constrZs(xs)
            for v in range(args.view_num):
                singleZs[v] += zs[v].cpu().tolist()
    return singleZs


def get_fea_com(fea):  # Get the common feature representation from view-specific features
    ZH_spec = []
    fea = [torch.tensor(item).to(device) for item in fea]
    with torch.no_grad():
        miss_vecs = [item.to(device) for item in Miss_vecs]
        for v in range(args.view_num):
            ZH_spec.append((torch.mul(fea[v].t(), miss_vecs[v].t()).float()))
    z_com = (torch.mul(sum(ZH_spec), (1 / sum(miss_vecs))).t()).float()
    z_com = z_com.cpu().tolist()
    return np.array(z_com)

def get_TotalResult_Comfea(fea_list, clustering_tool, y_truth):
    clustering_tool.fit(fea_list)
    y_pred = clustering_tool.labels_
    acc, nmi, purity, fscore, precision, recall, ari = evaluate(y_truth, y_pred)

    result_dic = dict(
            {'Missrate': mr, 'Epoch': args.epochs, 'Result_class': 'Pre_train', 'ACC': acc, 'NMI': nmi, 'PUR': purity,
             'Fscore': fscore,
             'Prec': precision, 'Recall': recall, 'ARI': ari})
    return result_dic


def get_MGCResult_spec(y):
    Assign = [[] for _ in range(args.view_num)]
    with torch.no_grad():
        for batch_idx, (xs, _, miss_vecs, _) in enumerate(All_cluster_loader):
            xs = [item.to(device) for item in xs]
            miss_vecs = [item.to(device) for item in miss_vecs]
            assign_result = model.Clustering(xs, miss_vecs)
            for v in range(args.view_num):
                Assign[v] += assign_result[v].cpu().tolist()
    Assign = [torch.tensor(a) for a in Assign]
    result = []
    for v in range(args.view_num):
        valid_indices = torch.nonzero(Miss_vecs[v].squeeze() != 0, as_tuple=False).squeeze(dim=1)
        Assign_valid = Assign[v][valid_indices]
        y_pred = np.array(torch.argmax(Assign_valid, dim=1))
        y_truth = y[valid_indices]
        acc, nmi, purity, fscore, precision, recall, ari = evaluate(y_truth, y_pred)
        result_dic = dict({'ACC': acc, 'NMI': nmi, 'PUR': purity, 'Fscore': fscore,
                       'Prec': precision, 'Recall': recall, 'ARI': ari})
        result.append(result_dic)
    return result


def pretrain(Epochs):
    Pre_Dataset = TrainDataset_All(X, Y, Miss_vecs, idxs)
    Pre_Loader = torch.utils.data.DataLoader(dataset=Pre_Dataset, batch_size=args.batch_pre, shuffle=True, drop_last=False)  # For small datasets, set drop_last=False
    t_progress = tqdm(range(Epochs), desc='Pretraining')
    for epoch in t_progress:
        tot_l_recon = 0.0
        for batch_idx, (xs, _, miss_vec, _) in enumerate(Pre_Loader):
            xs = [item.to(device) for item in xs]
            miss_vec = [item.to(device) for item in miss_vec]
            loss_recon = model.train_Recon(xs, miss_vec)
            optimizer_pre.zero_grad()
            loss_recon.backward()
            optimizer_pre.step()
            tot_l_recon += loss_recon.item()

    print('Epoch {}'.format(epoch), 'loss_recon:{:.6f}'.format(tot_l_recon / len(Pre_Loader)))


def train(Epochs):
    torch.autograd.set_detect_anomaly(True)
    train_Dataset = TrainDataset_All(X, Y, Miss_vecs, idxs)
    train_Loader = torch.utils.data.DataLoader(dataset=train_Dataset, batch_size=args.batch, shuffle=True,
                                               drop_last=False)
    t_progress = tqdm(range(Epochs), desc='training')
    for epoch in t_progress:
        h_fea = get_SingleHs(train_Loader)
        z_fea = get_SingleZs(train_Loader)
        tmp_commonH = get_fea_com(h_fea)
        tmp_commonZ = get_fea_com(z_fea)
        estimator.fit(tmp_commonH)
        estimator.fit(tmp_commonZ)
        centroids_H = estimator.cluster_centers_
        centroids_Z = estimator.cluster_centers_
        with torch.no_grad():
            C_H = torch.tensor(centroids_H, requires_grad=False).float().cuda()
            C_H = normalize(C_H, dim=1, p=2)
            model.clu_H_layer.data = C_H

            C_Z = torch.tensor(centroids_Z, requires_grad=False).float().cuda()
            C_Z = normalize(C_Z, dim=1, p=2)
            model.clu_Z_layer.data = C_Z
        T_loss = 0.0
        Recon_loss = 0.0
        Constra_loss =0.0
        MGC_loss = 0.0

        for batch_idx, (xs, _, miss_vecs, idx) in enumerate(train_Loader):
            xs = [item.to(device) for item in xs]
            miss_vecs = [item.to(device) for item in miss_vecs]

            loss_Recon = model.train_Recon(xs, miss_vecs)
            z_spec_all, loss_Constr = model.train_Constr(xs, miss_vecs)
            assign_result, loss_MGC = model.train_Graph(xs, miss_vecs)

            tot_loss = loss_Recon + loss_Constr + loss_MGC
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

            T_loss += tot_loss.item()
            Recon_loss += loss_Recon.item()
            Constra_loss += loss_Constr.item()
            MGC_loss += loss_MGC.item()

        if (epoch + 1) % 10 == 0:
            print('\n----------------loss---------------------')
            print('Epoch {}'.format(epoch + 1), 'T_loss:{:.6f}'.format(T_loss / len(train_Loader)))
            print('Epoch {}'.format(epoch + 1), 'Recon_loss:{:.6f}'.format(Recon_loss / len(train_Loader)))
            print('Epoch {}'.format(epoch + 1), 'Constra_loss:{:.6f}'.format(Constra_loss / len(train_Loader)))
            print('Epoch {}'.format(epoch + 1), 'MGC_loss:{:.6f}'.format(MGC_loss / len(train_Loader)))

            print("===================train results=========================")
            Zs = get_SingleZs(All_cluster_loader)
            commonZ = get_fea_com(Zs)
            res_dic_cr = get_TotalResult_Comfea(commonZ, estimator, Y[0])
            print('Final Result: ACC=%.4f, NMI=%.4f, ARI=%.4f' % (res_dic_cr['ACC'], res_dic_cr['NMI'], res_dic_cr['ARI']))
            write_dic_cr = dict(
                {'Missrate': mr, 'Epoch': (epoch + 1), 'Result_class': 'training test', 'ACC': res_dic_cr['ACC'],
                 'NMI': res_dic_cr['NMI'], 'PUR': res_dic_cr['PUR'], 'Fscore': res_dic_cr['Fscore'],
                 'Prec': res_dic_cr['Prec'], 'Recall': res_dic_cr['Recall'], 'ARI': res_dic_cr['ARI']})
            All_Metrics.append(write_dic_cr)
            pd_toExcel(All_Metrics, file_name)


if __name__=='__main__':
    try:
        dataset = {
            0: "NUS-WIDE-OBJECT-10",
        }
        now_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        txt_name = "./logs/" + dataset[0] + '_' + now_str + ".txt"
        logger = Logger(txt_name)
        sys.stdout = logger  # Redirect stdout to logger
        print(txt_name)
        file_name = "./logs/" + dataset[0] + '_' + now_str + ".xlsx"
        print(file_name)
        # Load hyperparameters
        with open('config.yaml', 'r', encoding='utf-8') as f:
            hyperparams = yaml.safe_load(f)
        for data_id in dataset:
            dataset_name = dataset[data_id]
            params = hyperparams.get(dataset_name, {})
            All_Metrics = []
            for mr in [0.1, 0.3,0.5,0.7]:
                print(dataset_name)

                print('--------------------Missing rate = ' + str(mr) + '--------------------')
                seed = params.get('seed', 42)
                seed_everything(seed)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                X, Y, Filled_X_com, Miss_vecs, cluster_num, data_num, view_num, view_dims = load_data(dataset_name, mr)
                idxs = np.array([i for i in range(data_num)])


                parser = argparse.ArgumentParser(description='train')
                parser.add_argument('--device', default=device)
                parser.add_argument('--dataset', default=str(data_id), help='name of dataset')
                parser.add_argument('--cluster_num', default=cluster_num, type=int, help='number of clusters')
                parser.add_argument('--data_num', default=data_num, type=int, help='number of samples')
                parser.add_argument('--view_num', default=view_num, type=int, help='number of views')
                parser.add_argument('--view_dims', default=view_dims, type=int, help='dimension of views')

                # Basic_params:
                parser.add_argument('--missrate', default=mr, type=float, help='missing rate of multi-view data')
                parser.add_argument('--Pre_epochs', default=50, type=int, help='epoches of Pretraining')
                parser.add_argument('--epochs', default=100, type=int, help='epochs of training')
                parser.add_argument('--batch_pre', default=512, type=int, help='batch size of Pretraining stage')
                parser.add_argument('--batch', default=512, type=int, help='batch size of training stage') # Recommended range:  512 1024
                parser.add_argument('--lr_pre', default=0.0003, type=float, help='learning rate of Pretraining stage') # Recommended range: 0.0003–0.001
                parser.add_argument('--lr_train', default=0.0005, type=float, help='learning rate of training stage') # Recommended range: 0.0003–0.001
                parser.add_argument('--recon_fea_dim', default=64, type=int, help='dimension of embedding')
                parser.add_argument('--gamma', default=1.0, type=float, help='compute_similarity_matrix with kernel function')
                parser.add_argument('--alpha', default=1.0, type=float, help='soft label hyper param')
                # Constr_params:
                parser.add_argument('--tau', default=0.2, type=float, help='temperature parameter in training loss') # Recommended range: 0.05-0.2
                parser.add_argument('--epsilon', default=0.05, type=float, help='regularization parameter for Sinkhorn-Knopp algorithm') # Recommended range: 0.03-0.1
                parser.add_argument('--sinkhorn_iterations', default=3, type=int, help='number of iterations in Sinkhorn-Knopp algorithm')
                parser.add_argument('--z_dim', default=64, type=int, help='dimension of contrastive learning embedding')
                # MGC_params:
                parser.add_argument('--K_neighber', default=3, type=int, help='dimension of views')
                parser.add_argument('--collapse_regularization', default=0.2, type=float, help='collapse regularization') # 0.01-0.3
                parser.add_argument('--lamda', default=0.1, type=float, help='lamda for Sinkhorn-Knopp algorithm')
                parser.add_argument('--graph_out_dim', default=64, type=int, help='dimension of graph out embedding')
                parser.add_argument('--graph_h_dim', default=128, type=int, help='dimension of graph hidden embedding')

                # If the dataset is YAML-specified, use YAML parameters to override defaults.
                if dataset_name in hyperparams:
                    params = hyperparams[dataset_name]
                    train_param = params.get('train_param', {})
                    constr_param = params.get('Constr_param', {})
                    mgc_param = params.get('MGC_param', {})

                    # Map YAML parameter names to argparse argument names
                    defaults_to_set = {
                        'lr_pre': train_param.get('lr_pre'),
                        'lr_train': train_param.get('lr_train'),
                        'batch': train_param.get('batch_size'),
                        'batch_pre': train_param.get('batch_size'),
                        'Pre_epochs': train_param.get('Pre_epochs'),
                        'epochs': train_param.get('epochs'),
                        'tau': constr_param.get('tau'),
                        'epsilon': constr_param.get('epsilon'),
                        'sinkhorn_iterations': constr_param.get('sinkhorn_iterations'),
                        'recon_fea_dim': constr_param.get('recon_dim'),
                        'z_dim': constr_param.get('z_dim'),
                        'K_neighber': mgc_param.get('K_neighber'),
                        'graph_h_dim': mgc_param.get('graph_h_dim'),
                        'collapse_regularization': mgc_param.get('collapse_regularization'),
                    }

                    defaults_to_set = {k: v for k, v in defaults_to_set.items() if v is not None}
                    parser.set_defaults(**defaults_to_set)

                args = parser.parse_args()

                print('param config:')
                print(f"  seed: {seed}")
                print(f"  lr_pre: {args.lr_pre}")
                print(f"  lr_train: {args.lr_train}")
                print(f"  batch_size: {args.batch}")
                print(f"  Pre_epochs: {args.Pre_epochs}")
                print(f"  epochs: {args.epochs}")
                print(f"  tau: {args.tau}")
                print(f"  epsilon: {args.epsilon}")
                print(f"  sinkhorn_iterations: {args.sinkhorn_iterations}")
                print(f"  recon_dim: {args.recon_fea_dim}")
                print(f"  z_dim: {args.z_dim}")
                print(f"  K_neighber: {args.K_neighber}")
                print(f"  graph_h_dim: {args.graph_h_dim}")
                print(f"  collapse_regularization: {args.collapse_regularization}")


                estimator = KMeans(n_clusters=args.cluster_num)
                model = FreeCSL(args).to(device)
                optimizer_pre = torch.optim.Adam(model.parameters(), lr=args.lr_pre)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_train)


                All_dataset = TrainDataset_All(X, Y, Miss_vecs, idxs)
                All_cluster_loader = torch.utils.data.DataLoader(dataset=All_dataset, batch_size=args.batch, shuffle=False, drop_last=False)

                #################################################
                # Pre-training
                #################################################
                pretrain(args.Pre_epochs)
                #################################################
                # training
                # ###################################################################
                train(args.epochs)
                print("===================Final results=========================")
                Zs = get_SingleZs(All_cluster_loader)
                commonZ = get_fea_com(Zs)
                res_dic_cr = get_TotalResult_Comfea(commonZ, estimator, Y[0])
                print('Final Result: ACC=%.4f, NMI=%.4f, ARI=%.4f' % (
                res_dic_cr['ACC'], res_dic_cr['NMI'], res_dic_cr['ARI']))
                write_dic_cr = dict(
                    {'Missrate': mr, 'Epoch':args.epochs, 'Result_class': 'final results', 'ACC': res_dic_cr['ACC'],
                     'NMI': res_dic_cr['NMI'], 'PUR': res_dic_cr['PUR'], 'Fscore': res_dic_cr['Fscore'],
                     'Prec': res_dic_cr['Prec'], 'Recall': res_dic_cr['Recall'], 'ARI': res_dic_cr['ARI']})
                All_Metrics.append(write_dic_cr)
                pd_toExcel(All_Metrics, file_name)
                # ##################################################

    finally:
        logger.log.close()