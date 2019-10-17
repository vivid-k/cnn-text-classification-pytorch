import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        self.emb_num = args.embed_num
        self.emb_dim = args.embed_dim
        self.class_num = args.class_num
        Ci = 1
        self.kernel = args.kernel_num
        self.Ks = args.kernel_sizes

        self.embed = nn.Embedding(self.emb_num, self.emb_dim)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, self.kernel, (K, self.emb_dim)) for K in self.Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(self.Ks)*self.kernel, self.class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D) 64*1*30*512
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
