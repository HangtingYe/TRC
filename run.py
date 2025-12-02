import torch
from torch.utils.data import TensorDataset, DataLoader
from data import load
from TRC import TRC
import random
import numpy as np
import json
import argparse
import sklearn
from sklearn.metrics import mean_squared_error
import scipy
import gc
import toml
import os
from backbone import *

def get_label(data_loader):
        y = []
        for _,_,y_ in data_loader:
            y.append(y_)
        y = torch.cat(y,dim=0).cpu().detach().numpy()
        return y
def get_pred_backbone(backbone,data_loader):
    backbone = backbone.eval()
    pred = []
    for i, (num_data, cat_data, target) in enumerate(data_loader):
        num_data, cat_data, target = num_data.cuda(), cat_data.cuda(), target.cuda()
        pred.append(backbone.predict(num_data, cat_data).data.cpu().numpy())
    return np.concatenate(pred, axis=0)

def get_pred_TRC(model_TRC,data_loader):
    model_TRC = model_TRC.eval()
    pred = []
    for i, (num_data, cat_data, target) in enumerate(data_loader):
        num_data, cat_data, target = num_data.cuda(), cat_data.cuda(), target.cuda()
        pred.append(model_TRC(num_data, cat_data).data.cpu().numpy())
    return np.concatenate(pred, axis=0)

def compute_score(pred, y, task_type, y_std):
    if task_type == 'binclass':
        pred = np.round(scipy.special.expit(pred))
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'multiclass':
        pred = pred.argmax(1)
        score = sklearn.metrics.accuracy_score(y.reshape(-1,1), pred.reshape(-1,1))
    elif task_type == 'regression':
        score = sklearn.metrics.mean_squared_error(y.reshape(-1,1), pred.reshape(-1,1)) ** 0.5 * y_std
    else:
        raise ValueError(f'task type {task_type} not supported')
    return score


def build_data_loader(dataset, batch_size=128, shuffle=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader



def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', type=str)
    parser.add_argument('--model_type', type=str)
    # parameter of TRC
    parser.add_argument('--shift_estimator', type=str, default='True',help='whether to implement tabular representation re-estimation')
    parser.add_argument('--space_mapping', type=str, default='True',help='whether to implement space compression')
    parser.add_argument('--loss_orth', type=str, default='True',help='whether to implement loss orth')
    parser.add_argument('--hyper_TRC', type=str, default='default')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    data_dir = './data'
    # config contains the hyperparameters for training the backbone model
    config = toml.load(f'./hyper/hyper_train/{args.model_type}.toml')
    # TRC_config contains the hyperparameters for training the TRC model
    TRC_config = toml.load(f'./hyper/hyper_TRC/{args.hyper_TRC}.toml')
    with open(f'{data_dir}/{args.dataname}/info.json') as f:
        info = json.load(f)
    _set_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    X, y, n_classes, y_mean, y_std, categories = load(args.dataname, info, config['data']['normalization'])
    task_type = info.get('task_type')
    print(task_type)

    n_num_features, n_cat_features = info.get('n_num_features'), info.get('n_cat_features')
    num_list = np.arange(n_num_features)
    cat_list = np.arange(n_num_features, n_num_features + n_cat_features) if n_cat_features!=None else None

    train_loader = build_data_loader(TensorDataset(X['train'][:,:n_num_features], X['train'][:,n_num_features:] if n_cat_features>0 else torch.empty(X['train'].shape[0], X['train'].shape[1]).cpu(), y['train']), config['training']['batch_size'], False)
    val_loader = build_data_loader(TensorDataset(X['val'][:,:n_num_features], X['val'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['val'].shape[0], X['val'].shape[1]).cpu(), y['val']), config['training']['batch_size'], False)
    test_loader = build_data_loader(TensorDataset(X['test'][:, :n_num_features], X['test'][:, n_num_features:] if n_cat_features>0 else torch.empty(X['test'].shape[0], X['test'].shape[1]).cpu(), y['test']), config['training']['batch_size'], False)

    

    print(config)
    ## model initialization
    model = Backbone(n_num_features, args.model_type, n_classes if task_type == 'multiclass' else 1, 
    info=info, config = config, categories = categories, seed=args.seed)

    if not os.path.exists('models'):
        os.mkdir('models')
    model = model.cuda()
    if os.path.exists(f'models/{args.dataname}_{args.model_type}_model.pth'):
        model.load_state_dict(torch.load(f'models/{args.dataname}_{args.model_type}_model.pth'))
    else:
        model.fit(train_loader, val_loader)
        torch.save(model.state_dict(), f'models/{args.dataname}_{args.model_type}_model.pth')


    pred = get_pred_backbone(model, test_loader)
    test_y = get_label(test_loader)
    baseline_score = compute_score(pred, test_y, task_type, y_std)
    print(f'baseline test score: {baseline_score}')


    
    _set_seed(args.seed)
    model = TRC(model, n_classes if task_type == 'multiclass' else 1, info, config, categories,TRC_config, args)
    model.cuda()
    model.fit(train_loader, val_loader)
    pred = get_pred_TRC(model, test_loader)
    test_y = get_label(test_loader)
    TRC_score = compute_score(pred, test_y, task_type, y_std)
    print(f'TRC test score: {TRC_score}')

    results = {
        "baseline_score":baseline_score,
        "TRC_score":TRC_score
    }
    import json
    
    if not os.path.exists('results'):
        os.mkdir('results')
    with open(f'results/{args.dataname}_{args.model_type}_{args.shift_estimator}_{args.space_mapping}_{args.loss_orth}_{args.seed}.json',"w") as f:
        json.dump(results, f, indent=4)
