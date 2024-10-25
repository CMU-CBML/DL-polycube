#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from torch_geometric.data import Data


# In[ ]:


rootdir = '..'
mesh_file_dir = rootdir + "/mesh_file"
text_file_dir = rootdir + "/text"

classification_res = np.loadtxt('../classification_result.txt')
classification_res = classification_res.astype('int')

output_size = np.loadtxt(rootdir + "/param/output_size.txt")
output_size = output_size.astype("int")


# In[ ]:


"""

"""
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_max_pool

class GCN_1(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN_1, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(9, hidden_channels)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 256)
        self.linear4 = Linear(128,256)   #
        self.linear2 = Linear(256,128)
        self.linear3 = Linear(128,128)
        self.lin = Linear(128, output_size[classification_res - 1])
        
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        

        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]


        
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.relu()
        x = self.lin(x)
        x = x.squeeze()
        return x

class GCN_2(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN_2, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(12, hidden_channels)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 256)
        self.linear4 = Linear(128,256) #
        self.linear2 = Linear(256,128)
        self.linear3 = Linear(128,128)
        self.lin = Linear(128, output_size[classification_res - 1])
        
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        

        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.relu()
        x = self.lin(x)
        x = x.squeeze()
        return x
    
    
    
class GCN_3(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN_3, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(12, hidden_channels)  
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 512)
        self.conv4 = GCNConv(512, 512)
        self.linear4 = Linear(128,256)  #
        self.linear2 = Linear(512,1024)
        self.linear3 = Linear(1024,512)
        self.lin = Linear(512, output_size[classification_res - 1])
 
        
    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        

        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        x = x.relu()
        x = self.lin(x)
        x = x.squeeze()
        return x

    
if classification_res == 2:
    model = GCN_1(hidden_channels=128)
elif classification_res in [3,4,5,6,7]:
    model = GCN_2(hidden_channels=128)
elif classification_res in [1,8,9,10]:
    model = GCN_3(hidden_channels=128)
print(model)  #


# In[ ]:


#

model_list = [
    "ring_800center_1920_12_e16c_warmup.ckpt",
    "longgenes1_800center_2_nostandardization.ckpt",
    "mount1_800center_1536_12_36c_warmup_new_scaled_v2.ckpt",
    "mount2_800center_2112_12_30c_warmup.ckpt",
    "ex1_800center_1760_12_e26c_warmup_v2.ckpt",
    "cat4_800center_4096_12_26c.ckpt",
    "cat5_800center_3024_12_30c.ckpt",
    "cow_800center_2432_12_62c_warmup_v4.ckpt",
    "fandisk_800center_2176_12_68c_warmup_v8.ckpt",
    "rockerarm_800center_3552_12_126c_warmup_new_v3_7_126.ckpt"
]

model.load_state_dict(torch.load("../model_parameter/" + model_list[classification_res - 1])) 
model.eval()


# torch.save(model.state_dict(), 'model.pth')
# 
# x_input = dataset[0]
# 
# 

# In[ ]:


"""

"""
feature = np.loadtxt(text_file_dir + '/testfeature.txt')
edge = np.loadtxt(text_file_dir + '/testedge.txt')

feature = torch.from_numpy(feature).to(torch.float32)[:,:]
edge = torch.from_numpy(edge).to(torch.int64).transpose(0,1)

meanfeat = torch.mean(feature[:,:9].reshape(-1,3,3),axis = 1).numpy()


data1 = Data(x = feature[:,:9],edge_index = edge) if classification_res == 2 else Data(x = feature[:,:],edge_index = edge)


# In[ ]:


import os
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
if(classification_res == 2) and (os.path.exists('../data/processed/' + test_name)):
    os.remove('../data/processed/' + test_name)
dataset = MyDataset(root = '../data/')


# In[ ]:


"""

"""
res1 = model(dataset[0].x,dataset[0].edge_index,torch.from_numpy(np.zeros(dataset[0].x.shape[0])).to(torch.int64))
print(res1)
resres = res1.detach().numpy()
r = resres.reshape(output_size[classification_res - 1] // 3,3)


# In[ ]:


"""

"""

_cats = np.loadtxt(text_file_dir + '/test_multi_cats_merged.txt')
_cats_centers = np.loadtxt(text_file_dir + '/test_centers.txt')
_cats_centers_id = np.loadtxt(text_file_dir + '/test_centers_id.txt').astype('int')

if classification_res == 1:
    numofmapping = [1,1,
                1,1,
                1,1,
                1,1,
                4,
                4]
elif classification_res == 2:
    numofmapping = [1,1,1,1,2,1,2,1,5,5]
elif classification_res == 3:
    numofmapping = [5,1,2,4,6,1,6,1,1,2,1,1,4,1]
elif classification_res == 4:
    numofmapping = [1,1,1,1,1,1,1,1,1,1,1,1,4,4,1,9]    
elif classification_res == 5:
    numofmapping = [1,2,1,1,2,1,1,2,1,1,2,1,1,4,1,4]
elif classification_res == 6:
    numofmapping = [1,1,1,2,1,3,1,3,1,2,4,5,1]    
elif classification_res == 7:
    numofmapping = [1,1,1,1,3,1,4,1,4,1,1,1,4,5,1]      
elif classification_res == 8:
    numofmapping = [1,7,1,1,8,1,1,1,8,1,1,1,8,1,1,10,1,1,1,1,1,5]       
elif classification_res == 9:
    numofmapping = [9,1,1,1,2,
                    4,4,
                    2,3,2,2,
                    2,7,
                    10,1,1,1,3,
                    12]
elif classification_res == 10:
    numofmapping = [3,2,1,9,1,1,1,
                   1,3,1,1,1,1,3,6,
                   1,2,1,3,2,1,2,2,6,
                   2,12,5,3,
                   1,19,2,2,
                   21,2,2]    
_cats_mapping = [list(range(sum(numofmapping[:i]),sum(numofmapping[:i]) + numofmapping[i])) for i in range(len(numofmapping))]


# In[ ]:


"""

"""

if classification_res == 2:
    tmp = r[5].copy()
    r[5] = r[6].copy()
    r[6] = tmp.copy()

    tmp = r[8].copy()
    r[8] = r[9].copy()
    r[9] = tmp.copy()
elif classification_res == 3:
    tmp0 = (r[2,2] + r[0,2]) / 2
    tmp1 = (r[1,1] + r[3,1] + r[4,1]) / 3
    tmp2 = (r[15,0] + r[22,0]) / 2
    tmp3 = (r[16,0] + r[23,0]) / 2
    tmp4 = (r[8,2] + r[10,2]) / 2

    r[0,2] = tmp0
    r[2,2] = tmp0
    r[1,1] = tmp1
    r[3,1] = tmp1
    r[4,1] = tmp1
    r[15,0] = tmp2
    r[22,0] = tmp2
    r[16,0] = tmp3
    r[23,0] = tmp3
    r[8,2] = tmp4
    r[10,2] = tmp4
elif classification_res == 4:
    tmp0 = (r[22,1] + r[24,1] + r[26,1] + r[28,1]) / 4
    tmp1 = (r[21,0] + r[23,0] + r[25,0] + r[27,0]) / 4
    tmp2 = (r[13,1] + r[15,1] + r[17,1] + r[19,1]) / 4
    tmp3 = (r[12,0] + r[14,0] + r[16,0] + r[18,0]) / 4

    r[22,1] = tmp0
    r[24,1] = tmp0
    r[26,1] = tmp0
    r[28,1] = tmp0

    r[21,0] = tmp1
    r[23,0] = tmp1
    r[25,0] = tmp1
    r[27,0] = tmp1

    r[29,0] = tmp1
    r[29,1] = tmp0

    r[13,1] = tmp2
    r[15,1] = tmp2
    r[17,1] = tmp2
    r[19,1] = tmp2

    r[12,0] = tmp3
    r[14,0] = tmp3
    r[16,0] = tmp3
    r[18,0] = tmp3
elif classification_res == 6:
    for k in range(2):
        r[k * 3 : k * 3 + 3,2] = _cats_centers[:3,2]
        r[k * 4 + 6 : k * 4 + 8,2] = _cats_centers[0,2]
        r[k * 4 + 8,2] = _cats_centers[1,2]
        r[k * 4 + 9,2] = _cats_centers[2,2]
elif classification_res == 7:
    mapping = [[0,1,2,3],[4,5,6,7],[8,9,10,11,12],[13,14,15,16,17],[18,19,20,21,22,23],[24,25,26,27,28,29]]
    K_means_cat = np.loadtxt(text_file_dir + '/Kmeans_mean.txt')
    mf = meanfeat[:,:3]
    tmp_cat = [i for i,x in enumerate(K_means_cat) if x == 0]
    epoch = 10
    u = r[mapping[0],:]
    for i in range(epoch):
        p = u[np.newaxis,:]
        p = p.repeat(len(tmp_cat),axis = 0)
        res = np.sum((mf[tmp_cat].reshape(len(tmp_cat),1,3) - p)**2,axis = 2)
        argres = np.argmin(res,axis = 1)
        #
        for i in range(4):
            u[i] = np.mean(mf[np.array(tmp_cat)[(argres == i)]],axis = 0)
    new_z = u
    for k in range(2):
        r[k*4:4*(k+1),2] = new_z[:,2]
        r[8 + k * 5,2] = new_z[0,2]
        r[8 + k * 5 + 1,2] = new_z[1,2]
        r[8 + k * 5 + 2,2] = new_z[1,2]
        r[8 + k * 5 + 3,2] = new_z[2,2]
        r[8 + k * 5 + 4,2] = new_z[3,2]
elif classification_res == 8:
    key_values = _cats_centers[[18,17,20,19,0,5,3,2]]
    r[[23,26,28,34,37,39,43,44,45,57],0] = key_values[0][0]
    r[[21,24,27,32,35,38,49,50,51,61],0] = key_values[2][0]

    mid_key_value_1 = (key_values[0][0] + key_values[2][0]) / 2

    r[[22,25,33,36,46,47,48,58,59,60],0] = mid_key_value_1

    r[42,0] = key_values[4][0] 

    r[[1,3,6,10,13,16,43,46,49,60],1] = key_values[0][1]
    r[[2,5,7,12,15,17,45,48,51,58],1] = key_values[1][1]

    mid_key_value_2 = (key_values[0][1] + key_values[1][1]) / 2

    r[[4,11,14,42,44,47,50,57,59,61],1] = mid_key_value_2


# In[ ]:


"""

"""
s_cat = [[],[]]


subcat = []
cat = []
accum_cat = 0
count = 0
for j in _cats_centers_id:
    print(j)
    tmp_cat = [i for i,x in enumerate(_cats) if x == j]
    cat.append(tmp_cat)
    p = r[_cats_mapping[count],:][np.newaxis,:]
    p = p.repeat(len(cat[count]),axis = 0)
    res = np.sum((meanfeat[cat[count]].reshape(len(cat[count]),1,3) - p)**2,axis = 2)
    argres = np.argmin(res,axis = 1)
    subcat.append(argres)
    
    for i in range(len(tmp_cat)):
        s_cat[0].append(tmp_cat[i])
    for i in range(len(argres)):
        s_cat[1].append(argres[i] + accum_cat)
    
    print("accum_cat",accum_cat)
    accum_cat += len(_cats_mapping[count])
    count += 1

s_cat = np.array(s_cat).T
u = s_cat[np.argsort(s_cat[:,0])][:,1]
np.savetxt(text_file_dir + '/test_allfaces.txt',u)
np.savetxt(text_file_dir + '/len_test.txt',np.array(accum_cat).reshape(1))

edge = edge.numpy().T


# In[ ]:


"""

"""

u = np.loadtxt(text_file_dir + '/test_allfaces.txt')
alist = [2,6,10,14]
new_edge = edge.reshape(int(edge.shape[0] / 3),-1)[:,[0,1,3,5]]
adj_cat = u[(new_edge)].astype('int')
t = u


epochs = 1
for epoch in range(epochs):
    adj_cat = u[(new_edge)].astype('int')
    for i in range(len(adj_cat)):
        if classification_res == 8:
            check = (np.argmax(np.bincount(adj_cat[i][1:])))
            if check != adj_cat[i][0]:
                u[i] = check
                adj_cat = u[(new_edge)].astype('int')
                #print("cat8",i,check)
        cnt = 0
        loc = 0
        for j in range(1,4):
            if adj_cat[i][0] != adj_cat[i][j]: 
                cnt += 1
            elif adj_cat[i][0] == adj_cat[i][j]:
                loc = new_edge[i][j]
        if cnt == 3:
            print("y",i)
            adj_cat[i][0] = np.argmax(np.bincount(adj_cat[i][1:]))
        if (classification_res not in [8,9]) and (cnt == 2):
            cnt2 = 0
            for j in range(1,4):
                if adj_cat[loc][0] != adj_cat[loc][j]:
                    cnt2 += 1
            if cnt2 == 2:
                print("isolated_diamond",i)
                tmp = np.argmax(np.bincount(adj_cat[i][1:]))
                adj_cat[i][0] = tmp
                adj_cat[loc][0] = tmp
        u = adj_cat[:,0]
np.savetxt(text_file_dir + '/newnew_test_allfaces.txt',u)


# In[ ]:




