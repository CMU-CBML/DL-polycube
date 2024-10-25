#!/usr/bin/env python
# coding: utf-8

# In[13]:


"""


"""


# In[14]:


import numpy as np
import os


rootdir = '..'
mesh_file_dir = rootdir + "/mesh_file"
text_file_dir = rootdir + "/text"


pattern1 = ['$# LS-DYNA Keyword file created by LS-PrePost(R) V4.5.3 - 28Oct2017',
'$# Created on Mar-15-2018 (10:40:36)',
'*KEYWORD',
'*ELEMENT_SHELL']
pattern2 = ['*NODE']
pattern3 = ['*END']

cat = np.loadtxt(text_file_dir + '/polygon_cat_res.txt')
cat = cat + 1

v_id = np.loadtxt(text_file_dir + '/v_id_per_face_polygon_res.txt').astype('int')
v_id = v_id + 1

faces_info = np.zeros((len(cat),10)).astype('int')
faces_info[:,0] = np.arange(1,len(cat) + 1)

faces_info[:,1] = cat
faces_info[:,2:5] = v_id
faces_info[:,5] = v_id[:,2]

faces_info = faces_info.astype('str')

for i in range(faces_info.shape[0]):
    for j in range(faces_info.shape[1]):
        faces_info[i][j] = faces_info[i][j].rjust(8)
        
np.savetxt(text_file_dir + '/faces_info_res.txt',faces_info,'%s',delimiter='')  

v_co = np.loadtxt(text_file_dir + '/v_co_polygon_res.txt')


p1 = (np.arange(1,len(v_co) + 1)).astype('int').reshape(-1,1).astype('str')
p2 = v_co.astype('str')
p3 = np.zeros((len(v_co),2)).astype('int').astype('str')

vertex_info = np.hstack((np.hstack((p1,p2)),p3))

for i in range(vertex_info.shape[0]):
    for j in range(vertex_info.shape[1]):
        vertex_info[i][j] = vertex_info[i][j].rjust(8) if (j == 0 or j == 4 or j == 5) else vertex_info[i][j].rjust(16)
        
np.savetxt(text_file_dir + '/vertices_info_res.txt',vertex_info,fmt = '%s',delimiter = '')


"""

"""
with open(text_file_dir + '/vertices_info_res.txt','r') as v_info:
    v_i = v_info.readlines()
with open(text_file_dir + '/faces_info_res.txt','r') as f_info:
    f_i = f_info.readlines()
    
    
"""

"""


output_file_name = mesh_file_dir + '/test_res.k'


if os.path.exists(output_file_name):
    os.remove(output_file_name)

if not os.path.exists(output_file_name):
    file = open(output_file_name, "w")
    file.close()

with open(output_file_name,'a+') as f:
    for i in range(len(pattern1)):
        f.write(pattern1[i])
        f.write('\n')
    for i in range(len(f_i)):
        f.write(f_i[i])
    for i in range(len(pattern2)):
        f.write(pattern2[i])
        f.write('\n')
    for i in range(len(v_i)):
        f.write(v_i[i])        
    for i in range(len(pattern3)):
        f.write(pattern3[i])
        f.write('\n')


# In[ ]:





# In[ ]:




