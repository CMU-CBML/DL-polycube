#!/usr/bin/env python
# coding: utf-8

# In[134]:


import torch
import numpy as np
from torch_geometric.data import Data


# In[135]:


import shutil
import os

rootdir = '..'
mesh_file_dir = rootdir + "/mesh_file"
text_file_dir = rootdir + "/text"

if os.path.exists(text_file_dir):
    pass
    #shutil.rmtree(text_file_dir)
else:
    os.mkdir(text_file_dir)


# In[136]:


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
        self.linear4 = Linear(128,256)
        self.linear2 = Linear(256,128)
        self.linear3 = Linear(128,128)
        self.lin = Linear(128, 11)
        
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        
        # 2. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
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


# In[137]:


"""

"""
feature = np.loadtxt(text_file_dir + '/testfeature.txt')
edge = np.loadtxt(text_file_dir + '/testedge.txt')


feature = torch.from_numpy(feature).to(torch.float32)
edge = torch.from_numpy(edge).to(torch.int64).transpose(0,1)

data1 = Data(x = feature,edge_index = edge)


# In[138]:


"""

"""

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


test_name = "test_data.pt"


class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [test_name]   

    def download(self):
        pass 

    def process(self):
        data_list = []

        data = data1
        data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
import os

if os.path.exists('../data/processed/' + test_name):
    os.remove('../data/processed/' + test_name)
dataset = MyDataset(root = '../data/')


# In[1]:


"""

"""

bat = torch.from_numpy(np.zeros(len(feature))).to(torch.int64)
model.eval()
res = model(dataset[0].x,dataset[0].edge_index,torch.from_numpy(np.zeros(dataset[0].x.shape[0])).to(torch.int64))
res = res[:,[2,8,10,6,7,4,0,3,1,5,9]]
res = res.argmax(dim = 1)
res = int(res)
print(res)
np.savetxt("../classification_result.txt",np.array([res]),"%d")


# In[140]:


"""

"""
import math
np.set_printoptions(suppress=True)

u = model(dataset[0].x,dataset[0].edge_index,torch.from_numpy(np.zeros(dataset[0].x.shape[0])).to(torch.int64))

t = np.array(u.detach().numpy())

def softmax(t):
    exp_t = math.e ** t
    probability = exp_t / np.sum(exp_t) * 100
    return probability
def softmax2(t):
    exp_t = math.e ** t
    probability = exp_t / np.sum(exp_t)
    return probability

def softmax3(t):
    exp_t = math.e ** (t - np.max(t))
    probability = exp_t / np.sum(exp_t) * 100
    return probability
probability = np.around(softmax3(t),9)
probability = probability[:,[2,8,10,6,7,4,0,3,1,5,9]]
print(probability)


# In[ ]:





# In[ ]:




