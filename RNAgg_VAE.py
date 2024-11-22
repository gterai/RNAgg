import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#class MLP_VAE(nn.Module):
#
#    def __init__(self, input_dim, output_dim, d_rep, device='cpu'):
#        super().__init__()
#        self.device  = device
#        self.loss    = torch.nn.CrossEntropyLoss(reduction='none')
#        self.encoder = MLP_Encoder(input_dim, d_rep, device=device)
#        self.decoder = MLP_Decoder(output_dim, d_rep, device=device)
#        #self.regress = MLP_regress(d_rep, device=device)
#        
#    def reparameterize(self, mean, var):
#        z = mean + torch.sqrt(var) * torch.randn(mean.size()).to(self.device)
#        return z

class MLP_VAE(nn.Module):

    def __init__(self, input_dim, output_dim, d_rep, device='cpu'):
        super().__init__()
        self.device  = device
        self.loss    = torch.nn.CrossEntropyLoss(reduction='none')
        self.encoder = MLP_Encoder(input_dim, d_rep, device=device)
        self.decoder = MLP_Decoder(output_dim, d_rep, device=device)
        #self.regress = MLP_regress(d_rep, device=device)
        
    def reparameterize(self, mean, var):
        z = mean + torch.sqrt(var) * torch.randn(mean.size()).to(self.device)
        return z

class MLP_Encoder(nn.Module):
    def __init__(self, input_dim, d_rep, device='cpu'):
        super().__init__()
        self.device=device
        #self.l1 = nn.Linear(input_dim, 256)
        #self.l2 = nn.Linear(256, 128)
        #self.l_mean = nn.Linear(128, DREP)
        #self.l_var = nn.Linear(128, DREP)
        #self.l1 = nn.Linear(input_dim, 1024)
        #self.l2 = nn.Linear(1024, 512)
        #self.l_mean = nn.Linear(512, DREP)
        #self.l_var = nn.Linear(512, DREP)
        self.l1 = nn.Linear(input_dim, 4096)
        self.l2 = nn.Linear(4096, 2048)
        self.l3 = nn.Linear(2048, 516)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(516)
        #self.l1 = nn.Linear(input_dim, 10000)
        #self.l2 = nn.Linear(10000, 5000)
        #self.l3 = nn.Linear(5000, 516)
        self.l_mean = nn.Linear(516, d_rep)
        self.l_var = nn.Linear(516, d_rep)
        self.input_dim = input_dim
        
    def forward(self, x):
        s = x.shape
        #print(s)
        x = x.view(s[0], self.input_dim) #ここで次元を揃えている。
        #print(x.shape)
        #exit(0)
        h = self.l1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.l2(h)
        h = self.bn2(h)
        h = torch.relu(h)
        h = self.l3(h)
        h = self.bn3(h)
        h = torch.relu(h)
        mean = self.l_mean(h)
        var = self.l_var(h)
        var = F.softplus(var)
        return mean, var

class MLP_Decoder(nn.Module):
    def __init__(self, out_dim, d_rep, device='cpu'):
        super().__init__()
        self.device = device
        #self.l1 = nn.Linear(DREP, 128)
        #self.l2 = nn.Linear(128, 256)
        #self.out = nn.Linear(256, out_dim)
        #self.l1 = nn.Linear(DREP, 512)
        #self.l2 = nn.Linear(512, 4096)
        #self.out = nn.Linear(4096, out_dim)

        self.l1 = nn.Linear(d_rep, 516)
        self.l2 = nn.Linear(516, 2048)
        self.l3 = nn.Linear(2048, 4096)
        self.out = nn.Linear(4096, out_dim)
        self.bn1 = nn.BatchNorm1d(516)
        self.bn2 = nn.BatchNorm1d(2048)
        self.bn3 = nn.BatchNorm1d(4096)

        #self.l2 = nn.Linear(516, 5000)
        #self.l3 = nn.Linear(5000, 10000)
        #self.out = nn.Linear(10000, out_dim)
    def forward(self, x):
        h = self.l1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.l2(h)
        h = self.bn2(h)
        h = torch.relu(h)
        h = self.l3(h)
        h = self.bn3(h)
        h = torch.relu(h)
        y = self.out(h)
        return y

    

class MLP_VAE_REGRE(nn.Module):

    def __init__(self, input_dim, output_dim, d_rep, device='cpu'):
        super().__init__()
        self.device  = device
        self.loss    = torch.nn.CrossEntropyLoss(reduction='none')
        self.encoder = MLP_Encoder(input_dim, d_rep, device=device)
        self.decoder = MLP_Decoder(output_dim, d_rep, device=device)
        self.regress = MLP_regress(d_rep, device=device)
        
    def reparameterize(self, mean, var):
        z = mean + torch.sqrt(var) * torch.randn(mean.size()).to(self.device)
        return z


class MLP_regress(nn.Module):
    def __init__(self, d_rep, device='cpu'):
        super().__init__()
        self.device=device
        self.l1 = nn.Linear(d_rep, d_rep)
        self.bn1 = nn.BatchNorm1d(d_rep)
        self.l2 = nn.Linear(d_rep, d_rep)
        self.bn2 = nn.BatchNorm1d(d_rep)
        self.l3 = nn.Linear(d_rep, 1)
        
    def forward(self, x):
        s = x.shape
        h = self.l1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.l2(h)
        h = self.bn2(h)
        h = torch.relu(h)
        h = self.l3(h)
        return h # regression

