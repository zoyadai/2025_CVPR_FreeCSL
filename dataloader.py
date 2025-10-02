import numpy as np
import torch
import h5py
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import warnings
warnings.filterwarnings("ignore")


path = 'Dataset/'


class TrainDataset_withPredsPriors(torch.utils.data.Dataset):
    def __init__(self, X, Y, Prior_Z, Pred_list, Miss_list, Idxs):
        self.X = X
        self.Y = Y
        self.Prior_Z = Prior_Z
        self.Pred_list = Pred_list
        self.Miss_list = Miss_list
        self.Idxs = Idxs
        self.view_size = len(X)

    def __getitem__(self, index):
        return [self.X[i][index] for i in range(self.view_size)], \
               [self.Y[i][index] for i in range(self.view_size)], \
               [self.Prior_Z[i][index] for i in range(self.view_size)], \
               [self.Pred_list[i][index] for i in range(self.view_size)], \
               [self.Miss_list[i][index] for i in range(self.view_size)], \
               self.Idxs[index]

    def __len__(self):
        return self.X[0].shape[0]


class TrainDataset_withPreds(torch.utils.data.Dataset):
    def __init__(self, X, Y, Pred_list, Miss_list, Idxs):
        self.X = X
        self.Y = Y
        self.Pred_list = Pred_list
        self.Miss_list = Miss_list
        self.Idxs = Idxs
        self.view_size = len(X)

    def __getitem__(self, index):
        return [self.X[i][index] for i in range(self.view_size)], \
               [self.Y[i][index] for i in range(self.view_size)], \
               [self.Pred_list[i][index] for i in range(self.view_size)], \
               [self.Miss_list[i][index] for i in range(self.view_size)], \
               self.Idxs[index]

    def __len__(self):
        return self.X[0].shape[0]


class PreDataset_filled(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X
        self.view_size = len(X)

    def __getitem__(self, index):
        return [self.X[i][index] for i in range(self.view_size)]

    def __len__(self):
        # return the total size of data
        return self.X[0].shape[0]


class TrainDataset_All(torch.utils.data.Dataset):
    def __init__(self, X, Y, Miss_list, Idxs):
        self.X = X
        self.Y = Y
        self.Miss_list = Miss_list
        self.Idxs = Idxs
        self.view_size = len(X)

    def __getitem__(self, index):
        return [self.X[i][index] for i in range(self.view_size)], \
               [self.Y[i][index] for i in range(self.view_size)], \
               [self.Miss_list[i][index] for i in range(self.view_size)], \
               self.Idxs[index]

    def __len__(self):
        # return the total size of data
        return self.X[0].shape[0]


def get_mask(view_num, alldata_len, missing_rate):
    """
    Generate a missingness matrix.
    :param view_num: Number of views.
    :param alldata_len: Total number of data points.
    :param missing_rate: The rate of missingness.
    :return: A binary matrix of size (alldata_len, view_num) indicating missingness.
    """
    miss_mat = np.ones((alldata_len, view_num))  # 1 indicates complete, 0 indicates missing
    b = ((10 - 10 * missing_rate) / 10) * alldata_len
    miss_begin = int(b)  # Index from which to start introducing missingness
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num, size=view_num - 1)
        miss_mat[i, missdata] = 0  # Set missing views to 0

    miss_mat = torch.tensor(miss_mat, dtype=torch.int)

    return miss_mat, miss_begin  # The range for complete data is [0, miss_begin]


def load_data(data_name, missrate):
    ###### Load Data ######
    data = h5py.File(path + data_name + ".mat")
    X, Y = [], []
    Label = np.array(data['Y']).T
    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()

    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        ###### Normalize features ######
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)

    input_dims = [X[i].shape[1] for i in range(len(X))]

    ###### Get clustering parameters ######
    cluster_num = len(np.unique(Y[0]))  # Number of clusters
    data_num = len(Y[0])  # Number of samples
    view_num = len(X)  # Number of views
    view_dims = input_dims  # Dimensions of each view

    ###### Shuffle original data ######
    index = np.random.permutation(data_num)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]

    ###### Get missingness matrix ######
    miss_mat, miss_begin = get_mask(view_num, data_num, missrate)
    Miss_vecs = [row for row in miss_mat.T]

    ###### Resample and align ######
    com_idx = [[] for _ in range(view_num)]  # Indices of complete samples for each view
    miss_idx = [[] for _ in range(view_num)] # Indices of missing samples for each view
    for i in range(miss_mat.shape[0]):
        for j in range(view_num):
            if miss_mat[i, j] == 1:
                com_idx[j].append(i)  # The j-th view of the i-th sample is complete
            else:
                miss_idx[j].append(i)


    ### Pad the indices of complete samples for each view to be of the same length
    filled_com_idx = []
    max_len = max(len(idx) for idx in com_idx)
    for idx in com_idx:
        if len(idx) < max_len:
            diff_value = random.sample(idx, max_len - len(idx))
            filled_com_idx.append(idx + diff_value)
        else:
            filled_com_idx.append(idx)

    Filled_X_com = [X[v][filled_com_idx[v]] for v in range(view_num)]
    X = [torch.from_numpy(X[v]) for v in range(view_num)]
    Filled_X_com = [torch.from_numpy(Filled_X_com[v]) for v in range(view_num)]

    return X, Y, Filled_X_com, Miss_vecs, cluster_num, data_num, view_num, view_dims










