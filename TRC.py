import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
# from transformer import Net
import os
import copy
import random
from torch.nn.parallel import parallel_apply
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
    
class TRC(nn.Module):
    def __init__(self, encoder, out_dim, info, config, categories, TRC_config,args) -> None:
        super().__init__()
        

        self.out_dim = out_dim
        self.info = info
        task_type = info.get('task_type')
        self.loss_func = get_loss_func(task_type)
        self.num_list = np.arange(info.get('n_num_features'))
        self.cat_list = np.arange(info.get('n_num_features'), info.get('n_num_features') + info.get('n_cat_features')) if info.get('n_cat_features')!=None else None
        self.categories = categories
        self.embedding_num = TRC_config['embedding_num']
        self.args = args


        self.config = config
        self.TRC_config = TRC_config
        self.encoder = encoder
        self.hidden_dim = encoder.get_hidden_dim()
        self.build_TRC_model()

    def build_TRC_model(self):
        if self.args.shift_estimator == 'True':
            _set_seed(self.args.seed)
            self.shift_estimator = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),nn.ReLU(),nn.Linear(self.hidden_dim,self.hidden_dim))

        if self.args.space_mapping == 'True':
            _set_seed(self.args.seed)
            self.coordinate_estimator = nn.Linear(self.hidden_dim, self.embedding_num)
            self.beta_linear = nn.Linear(self.embedding_num,self.hidden_dim,bias=False)
            betas = np.random.randn(self.TRC_config['embedding_num'], self.hidden_dim).astype('float32')
            self.beta_linear.weight = nn.Parameter(torch.tensor(betas.T))
        self.head_TRC = nn.Linear(self.hidden_dim, self.out_dim)

    
    def forward(self, inputs_n, inputs_c):
        encoder_hid = self.encoder(inputs_n, inputs_c).detach()
        # representation re-estimation
        if self.args.shift_estimator == 'True':
            encoder_hid = encoder_hid - self.shift_estimator(encoder_hid)

        # space compression
        if self.args.space_mapping == 'False':
            pred = self.head_TRC(encoder_hid)
        elif self.args.space_mapping == 'True':
            r = self.coordinate_estimator(encoder_hid)
            r = F.softmax(r,dim=-1)
            hid = self.beta_linear(r)
            pred = self.head_TRC(hid)
        return pred
    
    

    def _get_gradient(self, X_n,X_c,y):
        def cal_gradient(X_n, X_c, y, model, config):
            X_n, X_c, y = X_n.cuda(), X_c.cuda(), y.cuda()
            optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
            optimizer.zero_grad()
            if model.loss_func == F.cross_entropy:
                loss = model.loss_func(model.encoder.predict(X_n.unsqueeze(0), X_c.unsqueeze(0)), y.unsqueeze(0))
            else:
                loss = model.loss_func(model.encoder.predict(X_n.unsqueeze(0), X_c.unsqueeze(0)), y.unsqueeze(0).reshape(-1,1))
            loss.backward(retain_graph=True)
            gradient_norm = []
            for name, param in model.named_parameters():
                if param.grad != None:
                    gradient_norm.append(torch.abs(param.grad).reshape(-1))
            return torch.mean(torch.concat(gradient_norm)).item()
        tuples = [(X_n[i], X_c[i], y[i], copy.deepcopy(self), self.config) for i in range(X_n.shape[0])]  
        grads = parallel_apply([cal_gradient for k in range(X_n.shape[0])], tuples)
        return torch.tensor(grads)
    
    # to get observation of optimal representation and distribution of all data
    def _get_observation_of_optimal_rep_and_distribution(self,data_loader):
        self.eval()
        # get data from data_loader
        losses = []
        X_n = []
        X_c = []
        y = []
        for bid, (Xn_, Xc_, y_) in enumerate(data_loader):

            Xn_, Xc_, y_ = Xn_.cuda(), Xc_.cuda(), y_.cuda()
            pred = self.encoder.predict(Xn_, Xc_)
            if self.loss_func == F.cross_entropy:
                loss = self.loss_func(pred, y_,reduction='none')
            else:
                loss = self.loss_func(pred, y_.reshape(-1,1),reduction='none')
            losses.append(loss.detach().cpu())
            X_n.append(Xn_.detach().cpu())
            X_c.append(Xc_.detach().cpu())
            y.append(y_.detach().cpu())
        losses = torch.cat(losses, dim=0).squeeze()

        X_n = torch.cat(X_n, dim=0)
        X_c = torch.cat(X_c, dim=0)
        y = torch.cat(y, dim=0)

        distribution = torch.cat([X_n,X_c],dim=1)
        sample_num = losses.shape[0]
        

        # sort by loss
        sort_arg = torch.argsort(losses)
        losses = losses[sort_arg]
        X_n = X_n[sort_arg]
        X_c = X_c[sort_arg]
        y = y[sort_arg]


        # select optimal representation from top 20% samples with lowest loss
        top20_num = int(losses.shape[0]*0.2)
        losses = losses[:top20_num]
        X_n = X_n[:top20_num]
        X_c = X_c[:top20_num]
        y = y[:top20_num]


        # chose samples with lowest gradient
        grads = self._get_gradient(X_n,X_c,y)
        sort_arg = torch.argsort(grads)
        
        grads = grads[sort_arg]
        X_n = X_n[sort_arg]
        X_c = X_c[sort_arg]
        y = y[sort_arg]

        selected_num = int(sample_num*self.TRC_config['tau'])
        selected_num = max(selected_num, 1)
        grads = grads[:selected_num]
        X_n = X_n[:selected_num].detach().cpu()
        X_c = X_c[:selected_num].detach().cpu()
        y = y[:selected_num].detach().cpu()
        optimal_sample_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_n,X_c,y),batch_size=int(self.config['training']['batch_size']),shuffle=True)
        return optimal_sample_loader, distribution


    def perturb(self, X_n:torch.tensor,X_c:torch.tensor, distribution:torch.tensor, ratio:float,perturb_times):
        n_num_features = X_n.shape[1]
        X = torch.concat([X_n,X_c],dim=1)
        corrupted_X = X.repeat(perturb_times,1)
        corrupted_X = corrupted_X.reshape(-1,*X.shape)
        #
        mask = torch.rand(X.shape)
        mask[mask < ratio] = 1
        mask[mask >= ratio] = 0
        for i in range(perturb_times):
            random_index = torch.randint(0,X.shape[0], mask.shape)
            sample_data = distribution[random_index,torch.arange(X.shape[1])]
            corrupted_X[i][mask == 0] = sample_data[mask == 0]
        return corrupted_X[:,:,:n_num_features],corrupted_X[:,:,n_num_features:]

    # only for optimal representation
    def shift_leaning(self,denoise_data, optimizer):
        self.train()
        total_loss = 0
        optimal_sample_loader = denoise_data['optimal_sample_loader']
        distribution = denoise_data['distribution']
        for i, (inputs_n, inputs_c, targets) in enumerate(optimal_sample_loader):
            loss = 0
            inputs_n, inputs_c, targets = inputs_n.cuda(),inputs_c.cuda(),targets.cuda()
            encoder_reps = self.encoder(inputs_n, inputs_c)
            
            # for clean sample, pred shift should be zero
            zero_shift = torch.zeros_like(encoder_reps)
            pred_zero_shift = self.shift_estimator(encoder_reps)
            loss += F.mse_loss(pred_zero_shift,zero_shift)

            
            # data augmentation(corruption) and denoise
            perturb_times = self.TRC_config['perturb_times']
            ratio = random.uniform(0.1, 0.3)
            Xn_corrupted, Xc_corrupted = self.perturb(inputs_n.cpu(),inputs_c.cpu(),distribution,ratio, perturb_times)
            Xn_corrupted, Xc_corrupted = Xn_corrupted.cuda(), Xc_corrupted.cuda()
            for i in range(perturb_times):
                shifted_encoder_reps = self.encoder(Xn_corrupted[i], Xc_corrupted[i])
                pred_shift = self.shift_estimator(shifted_encoder_reps)
                loss += F.mse_loss(pred_shift,shifted_encoder_reps-encoder_reps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(optimal_sample_loader)
    
    def _run_one_epoch(self, data_loader, denoise_data, optimizer):
        if optimizer is not None:
            self.train()
        else:
            self.eval()
        total_loss = 0
        denoise_loss = 0
        if self.args.shift_estimator == 'True' and optimizer is not None:
            denoise_loss += self.shift_leaning(denoise_data,optimizer)
        for i, (inputs_n, inputs_c, targets) in enumerate(data_loader):
            loss = 0
            inputs_n, inputs_c, targets = inputs_n.cuda(),inputs_c.cuda(),targets.cuda()
            
            pred = self.forward(inputs_n, inputs_c)
            if self.loss_func == F.cross_entropy:
                loss += self.loss_func(pred, targets)
            else:
                loss += self.loss_func(pred, targets.reshape(-1,1))

            # for orth regularization
            if self.args.space_mapping == 'True' and self.args.loss_orth == 'True':
                assert self.TRC_config['reg_weight'] > 0.0
                betas = self.beta_linear.weight.T
                r_1 = torch.sqrt(torch.sum(betas**2,dim=1,keepdim=True))
                beta_metrix = torch.mm(betas, betas.T) / torch.mm(r_1, r_1.T)
                beta_metrix = torch.clamp(beta_metrix.abs(), 0, 1)
                l1 = torch.sum(beta_metrix.abs())
                l2 = torch.sum(beta_metrix ** 2)
                loss_sparse = l1 / l2
                loss_constraint = torch.abs(l1 - beta_metrix.shape[0])
                r_loss = loss_sparse + 0.5*loss_constraint
                loss = loss + r_loss*self.TRC_config['reg_weight']
            total_loss += loss.item()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
        if optimizer is not None:
            return total_loss / len(data_loader), denoise_loss
        else:
            return total_loss / len(data_loader)

    def fit(self, train_loader, val_loader):

        parameters = list(self.head_TRC.parameters())
        if self.args.shift_estimator == 'True':
            parameters += list(self.shift_estimator.parameters())
        if self.args.space_mapping == 'True':
            parameters += list(self.coordinate_estimator.parameters())+list(self.beta_linear.parameters())
        optimizer = optim.AdamW(parameters, lr=self.config['training']['lr'], weight_decay=self.config['training']['weight_decay'])

        best_val_loss = float('inf')
        best_model = None
        best_loss_epoch = 0
        n_epochs = self.config['training']['n_epochs']
        patience = self.config['training']['patience']
        optimal_sample_loader, distribution = self._get_observation_of_optimal_rep_and_distribution(val_loader)
        denoise_data = {'optimal_sample_loader':optimal_sample_loader,'distribution':distribution}

        for epoch in range(n_epochs):
            train_loss,denoise_loss = self._run_one_epoch(train_loader, denoise_data, optimizer)
            val_loss = self._run_one_epoch(val_loader, denoise_data, None)
            
            print(f'Epoch {epoch} train loss: {train_loss}, val loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.state_dict())
                best_loss_epoch = epoch
           
            if epoch - best_loss_epoch >= patience:
                break
        self.load_state_dict(best_model)
        return best_val_loss
            
    
        
        