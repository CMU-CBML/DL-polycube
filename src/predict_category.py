#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""

"""


# In[1]:


import torch
import numpy as np
from torch_geometric.data import Data


# In[2]:


from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_max_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(12, hidden_channels)  
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 256)
        self.linear4 = Linear(128,256)   #
        self.linear2 = Linear(256,128)
        self.linear3 = Linear(128,128)
        self.lin = Linear(128, 11)

        
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        

        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.relu()
        x = self.lin(x)
        return x

model = GCN(hidden_channels=128)
print(model)

#
#
model.load_state_dict(torch.load("../model_parameter/800graphclassify_11cats_noRotation_based_on_polycube_hybrid_v3_wd401_09_epoch50.ckpt"))
model.eval()


# In[3]:


import math
np.set_printoptions(suppress=True)

def softmax3(t):
    exp_t = math.e ** (t - np.max(t))
    probability = exp_t / np.sum(exp_t) * 100
    return probability


# In[9]:


test_list = [
    "test_cube.pt",
    "test_ring.pt",
    "test_rod_v2.pt",
    "test_mount1.pt",
    "test_mount2.pt",
    "test_part.pt",
    "test_bust.pt",
    "test_duck_v2.pt",
    "test_cow.pt",
    "test_fandisk.pt",
    "test_rockerarm.pt",
]


# In[5]:


import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

res = []
for i in range(len(test_list)):
    class MyDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(MyDataset, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])


        @property
        def raw_file_names(self):
            return []

        @property
        def processed_file_names(self):
            return [test_list[i]]   

        def download(self):
            pass 

        def process(self):
            data_list = []
            data = data1
            data_list.append(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    dataset = MyDataset(root = '../data/')
    
    u = model(dataset[0].x,dataset[0].edge_index,torch.from_numpy(np.zeros(dataset[0].x.shape[0])).to(torch.int64))
    np.set_printoptions(suppress=True)
    t = np.array(u.detach().numpy())
    probability = np.around(softmax3(t),4)
    res.append(probability)


# In[6]:


res = np.array(res)
res = res.reshape(11,11)
res = res[:,[2,8,10,6,7,4,0,3,1,5,9]]
print(res)
print(np.argmax(res,axis = 1))


# In[8]:


import pandas as pd
pd.DataFrame(res).to_csv("../probability_distribution.csv")

