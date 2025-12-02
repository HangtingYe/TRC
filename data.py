import numpy as np
import sklearn.preprocessing
import torch
import os
import random
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def mask_num_value(N_train,N_val,N_test, ratio):#dtype = 'float'
    assert N_train.shape[1] == N_val.shape[1] == N_test.shape[1]
    train_size = N_train.shape[0]
    val_size = N_val.shape[0]
    num_list = np.arange(N_train.shape[1])

    
    # if mask ratio == 0.0, then do not mask any data
    if ratio == 0.0:
        return N_train, N_val, N_test
    
    # generate mask matrix
    X = np.concatenate([N_train, N_val, N_test], axis=0)
    mask = np.random.rand(*X.shape)
    X[mask < ratio] = np.nan
    
    # mask data of X
    for num_id in num_list:
        mean = np.nanmean(X[:train_size, num_id])
        X[np.isnan(X[:, num_id]), num_id] = mean

    return X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]

def mask_cat_value(C_train,C_val,C_test, ratio):#dtype = 'object'
    assert C_train.shape[1] == C_val.shape[1] == C_test.shape[1]
    train_size = C_train.shape[0]
    val_size = C_val.shape[0]
    cat_list = np.arange(C_train.shape[1])
    
    # if mask ratio == 0.0, then do not mask any data
    if ratio == 0.0:
        return C_train, C_val, C_test
    
    # generate mask matrix
    X = np.concatenate([C_train, C_val, C_test], axis=0)
    #some of the data is None, so we need to replace it with 'nan'
    X[X == None ] = 'nan'
    mask = np.random.rand(*X.shape)
    X[mask < ratio] = 'nan'
    return X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]

def swap_noise(X, ratio):
    distribution = X.copy()
    corrupted_X = X.copy()
    #
    mask = np.random.rand(*X.shape)
    mask[mask < ratio] = 0
    mask[mask >= ratio] = 1

    random_index = np.random.randint(0,X.shape[0], mask.shape)
    sample_data = distribution[random_index,np.arange(X.shape[1])]
    corrupted_X[mask == 0] = sample_data[mask == 0]
    return corrupted_X

def noise_value(train,val,test, ratio):
    assert train.shape[1] == val.shape[1] == test.shape[1]
    train_size = train.shape[0]
    val_size = val.shape[0]  
    if ratio == 0.0:
        return train, val, test
    X = np.concatenate([train, val, test], axis=0)
    corrupted_X = swap_noise(X,ratio)
    return corrupted_X[:train_size], corrupted_X[train_size:train_size+val_size], corrupted_X[train_size+val_size:]
    

def nan_to_mean(N_train,N_val,N_test):
    for i in range(N_train.shape[1]):
        mean = np.nanmean(N_train[:,i])
        N_train[:,i] = np.nan_to_num(N_train[:,i],nan=mean)
        N_val[:,i] = np.nan_to_num(N_val[:,i],nan=mean)
        N_test[:,i] = np.nan_to_num(N_test[:,i],nan=mean)
    return N_train,N_val,N_test

def load(dataname, info, normalization):

    task_type, n_num_features, n_cat_features = info.get('task_type'), info.get('n_num_features'), info.get('n_cat_features')

    assert task_type in ['binclass', 'multiclass', 'regression']

    
    data_dir = './data'
    ## numerical features
    N_train, N_val, N_test = np.load(f'{data_dir}/{dataname}/N_train.npy', allow_pickle=True).astype('float32'), np.load(f'{data_dir}/{dataname}/N_val.npy', allow_pickle=True).astype('float32'), np.load(f'{data_dir}/{dataname}/N_test.npy', allow_pickle=True).astype('float32')

    # process missing value
    N_train,N_val,N_test = nan_to_mean(N_train,N_val,N_test)

    train_size = N_train.shape[0]
    val_size = N_val.shape[0]
    test_size = N_test.shape[0]

    N = np.concatenate([N_train, N_val, N_test], axis=0).astype('float32')

    
    ### feature-wise normalize
    if normalization == 'standard':
        preprocess = sklearn.preprocessing.StandardScaler().fit(N[:train_size])
    elif normalization == 'minmax':
        preprocess = sklearn.preprocessing.MinMaxScaler().fit(N[:train_size])
    elif normalization == 'quantile':
        preprocess = sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit(N[:train_size])
    elif normalization == 'none':
        preprocess = sklearn.preprocessing.FunctionTransformer().fit(N[:train_size])
    ### N: (bs, cols)
    N = preprocess.transform(N)


    ## catergorical features
    if n_cat_features != 0:
        C_train, C_val, C_test = np.load(f'{data_dir}/{dataname}/C_train.npy', allow_pickle=True), np.load(f'{data_dir}/{dataname}/C_val.npy', allow_pickle=True), np.load(f'{data_dir}/{dataname}/C_test.npy', allow_pickle=True)
        C = np.concatenate([C_train, C_val, C_test], axis=0)
        C = [sklearn.preprocessing.LabelEncoder().fit_transform(C[:,i]).astype('int64').reshape(-1,1) for i in range(C.shape[1])]
        C = np.concatenate(C, axis=1)

    else:
        C = None


    ## label
    y_train, y_val, y_test = np.load(f'{data_dir}/{dataname}/y_train.npy', allow_pickle=True), np.load(f'{data_dir}/{dataname}/y_val.npy', allow_pickle=True), np.load(f'{data_dir}/{dataname}/y_test.npy', allow_pickle=True)
    
    Y = np.concatenate([y_train, y_val, y_test], axis=0)
    Y = np.squeeze(Y)
    ### regression
    if task_type == 'regression':
        Y = Y.astype('float32')
    ### classification
    else:
        Y = sklearn.preprocessing.LabelEncoder().fit_transform(Y).astype('int64')

    n_classes = int(max(Y)) + 1 if task_type == 'multiclass' else None
    ### !!! CRUCIAL for neural networks when solving regression problems !!!
    if task_type == 'regression':
        y_mean = Y[:train_size].mean().item()
        y_std = Y[:train_size].std().item()
        Y = (Y - y_mean) / y_std
    elif task_type == 'binclass':
        y_mean = y_std = None
        Y = Y.astype('float32')
    elif task_type == 'multiclass':
        y_mean = y_std = None
        Y = Y.astype('int64')

    # generate train, val, test
    X = {}
    y = {}

    if n_cat_features != 0:
        X_all = np.concatenate([N,C], axis=1)
        # categories = np.max(C, axis=0) + 1
        # leave one for masking with the last per cat
        categories = np.max(C, axis=0) + 2
    else:
        X_all = N
        categories = None

    X['train'], X['val'], X['test'] = X_all[:train_size], X_all[train_size:train_size+val_size], X_all[-test_size:]
    y['train'], y['val'], y['test'] = Y[:train_size], Y[train_size:train_size+val_size], Y[-test_size:]

    X = {k: torch.tensor(v, dtype=torch.float).cpu() for k, v in X.items()}
    y = {k: torch.tensor(v).cpu() for k, v in y.items()}

    return X, y, n_classes, y_mean, y_std, categories