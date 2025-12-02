import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy
import random
import Models
def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_loss_func(task_type):
    if task_type == 'regression': 
        return F.mse_loss
    elif task_type == 'binclass':
        return F.binary_cross_entropy_with_logits
    elif task_type == 'multiclass':
        return F.cross_entropy
    else:
        raise ValueError(f'Unknown task type: {task_type}, should be one of [regression, binclass, multiclass]')
class Backbone(nn.Module):
    def __init__(self, input_num, model_type, out_dim, info, config, categories, seed) -> None:
        super().__init__()
        
        self.input_num = input_num ## number of numerical features
        self.out_dim = out_dim
        self.model_type = model_type
        self.info = info
        task_type = info.get('task_type')

        self.loss_func = get_loss_func(task_type)

        self.num_list = np.arange(info.get('n_num_features'))
        self.cat_list = np.arange(info.get('n_num_features'), info.get('n_num_features') + info.get('n_cat_features')) if info.get('n_cat_features')!=None else None
        self.categories = categories
        
        self.config = config
        self.args = None
        self.loss_line = float('inf')
        self.build_model(seed)



    def build_model(self,seed):
        # set seed to make sure the same model initialization in experiments expecially for ablation study
        _set_seed(seed) 

        if self.model_type == 'SNN':
            self.encoder = Models.snn.SNN(self.input_num, self.config['model']['d_layers'], self.config['model']['dropout'], self.out_dim, self.categories, self.config['model']['d_embedding'])
            self.hidden_dim = self.config['model']['d_layers'][-1]    
            self.head_1 = nn.Linear( self.hidden_dim, self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)  

        elif self.model_type == 'FTTransformer':
            self.encoder = Models.fttransformer.FTTransformer(self.input_num, self.categories, True, self.config['model']['n_layers'], self.config['model']['d_token'],
                            self.config['model']['n_heads'], self.config['model']['d_ffn_factor'], self.config['model']['attention_dropout'], self.config['model']['ffn_dropout'], self.config['model']['residual_dropout'],
                            self.config['model']['activation'], self.config['model']['prenormalization'], self.config['model']['initialization'], None, None)
            self.hidden_dim = self.config['model']['d_token']
            self.head_1 = nn.Linear( self.hidden_dim, self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)  
            

        elif self.model_type == 'ResNet':
            self.encoder = Models.resnet.ResNet(self.input_num, self.categories, self.config['model']['d_embedding'], self.config['model']['d'], self.config['model']['d_hidden_factor'], self.config['model']['n_layers'],
                            self.config['model']['activation'], self.config['model']['normalization'], self.config['model']['hidden_dropout'], self.config['model']['residual_dropout'])
            self.hidden_dim = self.config['model']['d']
            self.head_1 = nn.Linear( self.hidden_dim, self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)  
        
        
        elif self.model_type == 'DCN2':
            self.encoder = Models.dcn2.DCN2(self.input_num, self.config['model']['d'], self.config['model']['n_hidden_layers'], self.config['model']['n_cross_layers'],
                            self.config['model']['hidden_dropout'], self.config['model']['cross_dropout'], self.out_dim, self.config['model']['stacked'], self.categories, self.config['model']['d_embedding'])
            self.hidden_dim = self.config['model']['d'] if self.config['model']['stacked'] else 2 * self.config['model']['d']
            self.head_1 = nn.Linear( self.hidden_dim, self.out_dim)
            self.head_2 = copy.deepcopy(self.head_1)  

    def get_hidden_dim(self):
        return self.hidden_dim
    def get_out_dim(self):
        return self.out_dim
    def forward(self, inputs_n, inputs_c):
        rep = self.encoder(inputs_n, inputs_c)
        return rep
    def predict(self, inputs_n, inputs_c):
        encoder_hid = self.encoder(inputs_n, inputs_c)
        pred = self.head_1(encoder_hid)
        return pred

    def _run_one_epoch(self, data_loader, optimizer):
        
        if optimizer is not None:
            self.train()
        else:
            self.eval()
        total_loss = 0
        
        # for every sample in the data_loader
        for i, (inputs_n, inputs_c, targets) in enumerate(data_loader):
            loss = 0
            inputs_n, inputs_c, targets = inputs_n.cuda(), inputs_c.cuda(), targets.cuda()
            pred = self.predict(inputs_n, inputs_c)

            if self.loss_func == F.cross_entropy:
                loss += self.loss_func(pred, targets)
            else:
                loss += self.loss_func(pred, targets.reshape(-1,1))
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)
    
    def fit(self, train_loader, val_loader):
        parameters = list(self.encoder.parameters())+list(self.head_1.parameters())
        optimizer = optim.AdamW(parameters, lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])
        best_val_loss = float('inf')
        best_model = None
        best_loss_epoch = 0
        n_epochs = self.config['training']['n_epochs']
        patience = self.config['training']['patience']
        for epoch in range(n_epochs):
            train_loss = self._run_one_epoch(train_loader, optimizer)
            val_loss = self._run_one_epoch(val_loader, None)
            print(f'Epoch {epoch} train loss: {train_loss}, val loss: {val_loss}')#, val score: {val_score}, test score: {test_score}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.state_dict())
                best_loss_epoch = epoch
            if epoch - best_loss_epoch >= patience:
                break
        self.load_state_dict(best_model)

        return best_val_loss