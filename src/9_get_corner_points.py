#!/usr/bin/env python
# coding: utf-8

# In[107]:


import pandas as pd
import numpy as np

rootdir = '..'
mesh_file_dir = rootdir + "/mesh_file"
text_file_dir = rootdir + "/text"

classification_res = np.loadtxt('../classification_result.txt')
classification_res = classification_res.astype('int')

"""

"""
template_list = [
    "ring_adjacent_areas_template.txt",
    "rod_adjacent_areas_template.txt",
    "mount1_adjacent_areas_template.txt",
    "mount2_adjacent_areas_template.txt",
    "part_adjacent_areas_template.txt",
    "bust_adjacent_areas_template.txt",
    "duck_adjacent_areas_template.txt",
    "cow_adjacent_areas_template.txt",
    "fandisk_adjacent_areas_template.txt",
    "rockerarm_adjacent_areas_template.txt"
]

adjacent_areas_template = np.loadtxt("../adjacent_relationships/" + template_list[classification_res - 1])
adjacent_areas_template = np.array(adjacent_areas_template).astype("int")


# In[109]:


def check(a,b,q):
    for k in range(len(q)):
        if (a in q[k]) and (b in q[k]):
            return True
    return False

with open(text_file_dir + '/test_corner_points_candidates.txt') as f:
    qu = f.read().splitlines()
for i in range(len(qu)):
    qu[i] = eval(qu[i])  

with open(text_file_dir + '/test_corner_points_candidates.txt') as f:
    p = f.read().splitlines()
for i in range(len(p)):
    p[i] = eval(p[i])  
    
num_of_link_edges = np.loadtxt(text_file_dir + '/num_of_link_edges.txt')
v_co = np.loadtxt(text_file_dir + '/v_co_polygon.txt')
p_template = p.copy()



def getdistance(a,b):
    return np.sum((a-b)**2)


"""

"""
for i in range(len(p)):
    current_adj = adjacent_areas_template[i]
    while(len(p[i]) > 4):
        mindis = 99999999
        mina = 0
        minb = 0
        flag = 0
        for j in range(len(p[i])):
            for k in range(j+1,len(p[i])):
                if check(int(p[i][j]),int(p[i][k]),np.array(qu)[adjacent_areas_template[i]]):
                    continue
                dis = getdistance(v_co[int(p[i][j])],v_co[int(p[i][k])])
                if dis < mindis:
                    mindis = dis
                    mina = j
                    minb = k
                if (classification_res != 6) and (check(int(p[i][j]),int(p[i][k]),np.array(qu)[list(set(range(len(adjacent_areas_template))) - set(adjacent_areas_template[i]) - set([i]))])):
                    mindis = dis
                    mina = j
                    minb = k
                    #if i == 1:
                    #    print("Priority:",int(p[i][j]),int(p[i][k]))
                    flag = 1
                    break
            if flag == 1:
                break            
        #if i == 57:
        #    print(p[i])    
        #    print(p[i][mina],p[i][minb])
        if(num_of_link_edges[p[i][mina]] > num_of_link_edges[p[i][minb]]):
            p[i].pop(minb)
        else:
            p[i].pop(mina)
            
q = np.array(p).astype('int')
q_ = q.copy()


# In[111]:


"""

"""
ConnectList = []
for i in range(len(adjacent_areas_template)):
    Current_Connection = []
    for j in range(adjacent_areas_template.shape[1]):
        Current_Connection.append([i,adjacent_areas_template[i][j]])
    ConnectList.append(Current_Connection)
ConnectList = np.array(ConnectList).reshape(-1,2)


# In[1]:


"""

"""
def sortVertsWithAngle(p,v_co):
    c1 = v_co[p][0]
    c2 = v_co[p][1]
    c3 = v_co[p][2]
    c4 = v_co[p][3]

    v1 = c2 - c1
    v2 = c3 - c1
    v3 = c4 - c1
    def get_angle(va,vb):
        ea = np.sum(va ** 2) ** 0.5
        eb = np.sum(vb ** 2) ** 0.5
        
        cos = np.dot(va,vb) / ea / eb
        cos = np.clip(cos,-1,1)
        return np.arccos(cos)
    a1 = get_angle(v1,v2)
    a2 = get_angle(v1,v3)
    a3 = get_angle(v2,v3)
    if np.max([a1,a2,a3]) == a1:
        return [p[0],p[1],p[3],p[2]]
    elif np.max([a1,a2,a3]) == a2:
        return [p[0],p[1],p[2],p[3]]
    else:
        return [p[0],p[2],p[1],p[3]] 

q_edge_check = q_.copy()
for i in range(len(q)):
    q_edge_check[i] = np.array(sortVertsWithAngle(q_edge_check[i],v_co)).astype('int')

check_edges = []
for i in range(q_edge_check.shape[0]):
    for j in range(4):
        if(i == 61):
            print(j,(j+1) % 4)
            print(q_edge_check[i][j],q_edge_check[i][(j+1)%4])
        check_edges.append([q_edge_check[i][j],q_edge_check[i][(j+1)%4]])
        check_edges.append([q_edge_check[i][(j+1)%4],q_edge_check[i][j]])
check_edges = np.array(check_edges)


# In[113]:


"""

"""
def ConnectPlanes(p1,p2):
    error_list = np.array([[-1,-1]])
    count = 0
    print("Error List:",error_list)
    
    while(len(set(p1) - set(p2)) > 2):
        print("#Not Connected Vertices:",len(set(p1) - set(p2)))
        if(len(set(p1) - set(p2)) == 3):
            mindis = 99999999
            mina = 0
            minb = 0
            commonset = set(p1) - (set(p1) - set(p2))
            p1_1st_point = list(commonset)[0]
            p2_1st_point = list(commonset)[0]
            for i in range(4):
                for j in range(4):
                    if np.any(np.all([int(p1[i]),int(p2[j])] == error_list,axis = 1)):
                        continue
                    if p1[i] in commonset:
                        continue
                    if p2[j] in commonset:
                        continue
                    dis = getdistance(v_co[int(p1[i])],v_co[int(p2[j])])
                    if dis < mindis:
                        mindis = dis
                        mina = i
                        minb = j

            p1_2nd_point = int(p1[mina])
            p2_2nd_point = int(p2[minb])

        
            print("w",np.any(np.all([p1_1st_point,p1_2nd_point] == check_edges,axis = 1)),np.any(np.all([p2_1st_point,p2_2nd_point] == check_edges,axis = 1)))

            if ((np.any(np.all([p1_1st_point,p1_2nd_point] == check_edges,axis = 1))) and (np.any(np.all([p2_1st_point,p2_2nd_point] == check_edges,axis = 1)))) == False:
                print("EDGE ERROR")
                error_list = np.vstack([error_list,[p1_2nd_point,p2_2nd_point]])
                error_list = np.vstack([error_list,[p2_2nd_point,p1_2nd_point]])
                print(error_list)
                continue
            if classification_res == 6:
                q_[q_ == p1_2nd_point] = p2_2nd_point
                check_edges[check_edges == p1_2nd_point] = p2_2nd_point
                error_list[error_list== p1_2nd_point] = p2_2nd_point
            else:
                if(num_of_link_edges[p2_2nd_point] > num_of_link_edges[p1_2nd_point]):    
                    q_[q_ == p1_2nd_point] = p2_2nd_point
                    check_edges[check_edges == p1_2nd_point] = p2_2nd_point
                    error_list[error_list== p1_2nd_point] = p2_2nd_point
                    print(p1_2nd_point," out",p2_2nd_point," stay")
                else:
                    q_[q_ == p2_2nd_point] = p1_2nd_point  
                    check_edges[check_edges == p2_2nd_point] = p1_2nd_point
                    error_list[error_list== p2_2nd_point] = p1_2nd_point
                    print(p2_2nd_point," out",p1_2nd_point," stay")
        else:
            mindis = 99999999
            mina = 0
            minb = 0
            commonset = set(p1) - (set(p1) - set(p2))
            for i in range(4):
                for j in range(4):
                    if np.any(np.all([int(p1[i]),int(p2[j])] == error_list,axis = 1)):
                        continue
                    if p1[i] in commonset:
                        continue
                    if p2[j] in commonset:
                        continue
                    dis = getdistance(v_co[int(p1[i])],v_co[int(p2[j])])
                    if dis < mindis:
                        mindis = dis
                        mina = i
                        minb = j
            p1_1st_point = int(p1[mina])
            p2_1st_point = int(p2[minb])
            
            mindis = 99999999
            _2nd_mina = 0
            _2nd_minb = 0
            for i in range(4):
                for j in range(4):
                    if (i == mina) or (j == minb):
                        continue
                    if np.any(np.all([int(p1[i]),int(p2[j])] == error_list,axis = 1)):
                        continue
                    if p1[i] in commonset:
                        continue
                    if p2[j] in commonset:
                        continue
                    dis = getdistance(v_co[int(p1[i])],v_co[int(p2[j])])
                    if dis < mindis:
                        mindis = dis
                        _2nd_mina = i
                        _2nd_minb = j
            
            p1_2nd_point = int(p1[_2nd_mina])
            p2_2nd_point = int(p2[_2nd_minb])
            print("w",np.any(np.all([p1_1st_point,p1_2nd_point] == check_edges,axis = 1)),np.any(np.all([p2_1st_point,p2_2nd_point] == check_edges,axis = 1)))
            if ((np.any(np.all([p1_1st_point,p1_2nd_point] == check_edges,axis = 1))) and (np.any(np.all([p2_1st_point,p2_2nd_point] == check_edges,axis = 1)))) == False:
                print("EDGE ERROR")
                error_list = np.vstack([error_list,[p1_2nd_point,p2_2nd_point]])
                error_list = np.vstack([error_list,[p2_2nd_point,p1_2nd_point]])
                print(error_list)
                continue
            if classification_res == 6:
                q_[q_ == p1_1st_point] = p2_1st_point  
                check_edges[check_edges == p1_1st_point] = p2_1st_point
                error_list[error_list== p1_1st_point] = p2_1st_point

                q_[q_ == p1_2nd_point] = p2_2nd_point
                check_edges[check_edges == p1_2nd_point] = p2_2nd_point
                error_list[error_list== p1_2nd_point] = p2_2nd_point
            else:
                if(num_of_link_edges[p2_1st_point] > num_of_link_edges[p1_1st_point]):
                    q_[q_ == p1_1st_point] = p2_1st_point  
                    check_edges[check_edges == p1_1st_point] = p2_1st_point
                    error_list[error_list== p1_1st_point] = p2_1st_point
                    print(p1_1st_point," out",p2_1st_point," stay")
                else:
                    q_[q_ == p2_1st_point] = p1_1st_point  
                    check_edges[check_edges == p2_1st_point] = p1_1st_point
                    error_list[error_list== p2_1st_point] = p1_1st_point
                    print(p2_1st_point," out",p1_1st_point," stay")
                if(num_of_link_edges[p2_2nd_point] > num_of_link_edges[p1_2nd_point]):    
                    q_[q_ == p1_2nd_point] = p2_2nd_point
                    check_edges[check_edges == p1_2nd_point] = p2_2nd_point
                    error_list[error_list== p1_2nd_point] = p2_2nd_point
                    print(p1_2nd_point," out",p2_2nd_point," stay")
                else:
                    q_[q_ == p2_2nd_point] = p1_2nd_point  
                    check_edges[check_edges == p2_2nd_point] = p1_2nd_point
                    error_list[error_list== p2_2nd_point] = p1_2nd_point
                    print(p2_2nd_point," out",p1_2nd_point," stay")
        print("a ",p1_1st_point,p2_1st_point)
        print("b ",p1_2nd_point,p2_2nd_point)
    return p1,p2


# In[114]:


"""

"""
for i in range(len(ConnectList)):
    flag = 0
    print("Current Epoch:",i)
    print("Now Connecting ",ConnectList[i][0],ConnectList[i][1])
    print("Current Vertices of Plane A,B",list(q_[ConnectList[i][0]]),list(q_[ConnectList[i][1]]))
    ConnectPlanes(q_[ConnectList[i][0]],q_[ConnectList[i][1]])
    for j in range(len(q_)):
        if len(np.unique(q_[j])) < 4:
            flag = 1 
            print("Duplicate points：",j,q_[j])
    if flag == 1:
        break

for i in range(len(q_)):
    q_[i] = np.array(sortVertsWithAngle(q_[i],v_co)).astype('int')
            
            
np.savetxt(text_file_dir + '/test_corner_points.txt',q_,'%d')


# In[ ]:




