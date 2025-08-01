import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, num_classes=1, hidden_size=128, learning_rate=0.001, batch_size=25, mode='train'):
        super(Model, self).__init__()
        self.learning_rate = learning_rate
        self.hidden_repr_size = hidden_size
        self.batch_size = batch_size
        self.test_trials = 20
        self.num_classes = num_classes
        self.mode = mode

        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        
        self.flattened_dim = 64 * 56 * 56
        self.fc1 = nn.Linear(self.flattened_dim, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, num_classes)      
        self.fc_logvar = nn.Linear(hidden_size, num_classes)   

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        logits = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return logits, log_var, logits 

    def mc_forward(self, x):
      
        was_training = self.training
        self.train()

        probs_list = []
        for _ in range(self.test_trials):
            logits, _, _ = self.forward(x)
            probs = torch.sigmoid(logits)  
            probs_list.append(probs.unsqueeze(0))  

     
        if not was_training:
            self.eval()

        probs_stack = torch.cat(probs_list, dim=0)  
        mc_mean = probs_stack.mean(dim=0)           
        var_epistemic = probs_stack.var(dim=0)      
        return mc_mean, var_epistemic

    def build_train_loss(self, labels, logits, log_var):
        labels = labels.float().unsqueeze(1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        probs = torch.sigmoid(logits)
        precision = torch.exp(-log_var)
        loss1 = torch.mean(precision * (labels - probs) ** 2)
        loss2 = torch.mean(log_var)
        return 0.5 * (loss1 + loss2), loss1, loss2

    def build_test_outputs(self, x):
     
        mean_mc, var_epistemic = self.mc_forward(x)

        self.eval()
        with torch.no_grad():
            rec_mean, log_var, logits = self.forward(x)
            var_aleatoric = torch.exp(log_var)
            probs = torch.sigmoid(logits) 
            preds = (probs > 0.5).long().squeeze(1) 

     
        norm = torch.max(var_epistemic.mean(), var_aleatoric.mean())
        var_epistemic /= norm
        var_aleatoric /= norm

        return {
            'rec_mean': rec_mean,
            'epistemic_var': var_epistemic,
            'aleatoric_var': var_aleatoric,
            'mc_mean': mean_mc,
            'l2_error': (rec_mean - mean_mc) ** 2,
            'preds': preds,
            'probs': probs.squeeze(1) 
        }

    def configure_optimizer(self, lr):
        return optim.Adam(self.parameters(), lr)
