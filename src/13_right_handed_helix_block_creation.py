#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import os

rootdir = '..'
mesh_file_dir = rootdir + "/mesh_file"
text_file_dir = rootdir + "/text"

classification_res = np.loadtxt('../classification_result.txt')
classification_res = classification_res.astype('int')

corner_points_group = np.loadtxt(text_file_dir + '/test_corner_points.txt')
corner_points_group = corner_points_group.astype("int")


v_co = np.loadtxt(text_file_dir + '/v_co_polygon_res.txt')


# In[34]:


def get_common_vertices(p1,p2):
    return list(set(p1) - (set(p1) - set(p2)))

def get_single_vertex(p1,p2,p3):
    return get_common_vertices(get_common_vertices(corner_points_group[p1],corner_points_group[p2]),corner_points_group[p3])
def create_inner_vertex(v_co,v1,v2):
    new_v = [0,0,0]
    #print(v1,v2)
    new_v[0] = v_co[v1[0]][0]
    new_v[1] = v_co[v1[0]][1]
    new_v[2] = v_co[v2[0]][2]
    return new_v
def get_new_plane(p1,p2,p3,p4):
    return get_common_vertices(p1,p2) + get_common_vertices(p3,p4)

'''

'''
def Find_plane_equation(xo1, yo1, zo1, xo2, yo2, zo2, xo3, yo3, zo3):
    a = (yo2 - yo1) * (zo3 - zo1) - (zo2 - zo1) * (yo3 - yo1)
    b = (xo3 - xo1) * (zo2 - zo1) - (xo2 - xo1) * (zo3 - zo1)
    c = (xo2 - xo1) * (yo3 - yo1) - (xo3 - xo1) * (yo2 - yo1)
    d = -(a * xo1 + b * yo1 + c * zo1)
    Equation_param = np.array([a, b, c, d])
    return Equation_param
'''

'''
def Find_intersection(x1, y1, z1, x2, y2, z2, a, b, c, d):
    p = x1 - x2
    q = y1 - y2
    r = z1 - z2
    t = (a * x1 + b * y1 + c * z1 + d) / (a * p + b * q + c * r) * (-1)
    x = p * t + x1
    y = q * t + y1
    z = r * t + z1
    res = np.array([x, y, z])
    return res
"""

"""

def create_inner_vertex_with_lines_and_planes(vo1_p1,vo1_p2,vo1_p3,vo2_p1,vo2_p2,vo2_p3,vo3_p1,vo3_p2,vo3_p3,v1_p1,v1_p2,v1_p3,v2_p1,v2_p2,v2_p3):
    vo1 = v_co[get_single_vertex(vo1_p1,vo1_p2,vo1_p3)][0]
    vo2 = v_co[get_single_vertex(vo2_p1,vo2_p2,vo2_p3)][0]
    vo3 = v_co[get_single_vertex(vo3_p1,vo3_p2,vo3_p3)][0]
    
    v1 = v_co[get_single_vertex(v1_p1,v1_p2,v1_p3)][0]
    v2 = v_co[get_single_vertex(v2_p1,v2_p2,v2_p3)][0]
    
    plane_param = Find_plane_equation(vo1[0],vo1[1],vo1[2],vo2[0],vo2[1],vo2[2],vo3[0],vo3[1],vo3[2])
    return Find_intersection(v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],plane_param[0],plane_param[1],plane_param[2],plane_param[3])


# In[35]:


"""

"""
plane_groups = []
refer = []

if classification_res == 1:
    plane_groups = [[corner_points_group[12],corner_points_group[8]],
                    [corner_points_group[13],corner_points_group[9]],
                    [corner_points_group[14],corner_points_group[10]],
                    [corner_points_group[15],corner_points_group[11]]
                   ]
    
    refer = [get_common_vertices(corner_points_group[0],corner_points_group[6]),
             get_common_vertices(corner_points_group[2],corner_points_group[6]),
             get_common_vertices(corner_points_group[4],corner_points_group[2]),
             get_common_vertices(corner_points_group[0],corner_points_group[4])
            ]
elif classification_res == 2:
    plane_groups = [[corner_points_group[15],corner_points_group[10]],
                    [corner_points_group[18],corner_points_group[11]],
                    [corner_points_group[17],corner_points_group[12]],
                    [corner_points_group[16],corner_points_group[13]],
                    [corner_points_group[19],corner_points_group[14]],
                   ]

    refer = [get_common_vertices(corner_points_group[7],corner_points_group[8]),
             get_common_vertices(corner_points_group[7],corner_points_group[8]),
             get_common_vertices(corner_points_group[4],corner_points_group[5]),
             get_common_vertices(corner_points_group[0],corner_points_group[7]),
             get_common_vertices(corner_points_group[7],corner_points_group[8]),
            ]
elif classification_res == 3:
    extra_corner_points_group = []

    extra_plane1 = get_new_plane(corner_points_group[3],corner_points_group[4],corner_points_group[11],corner_points_group[27])
    extra_plane2 = get_new_plane(corner_points_group[6],corner_points_group[7],corner_points_group[5],corner_points_group[28])
    extra_plane3 = get_new_plane(corner_points_group[0],corner_points_group[1],corner_points_group[8],corner_points_group[9])
    extra_plane4 = get_new_plane(corner_points_group[0],corner_points_group[3],corner_points_group[8],corner_points_group[11])
    extra_plane5 = get_new_plane(corner_points_group[1],corner_points_group[2],corner_points_group[9],corner_points_group[10])
    extra_plane6 = get_new_plane(corner_points_group[2],corner_points_group[3],corner_points_group[10],corner_points_group[11])

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)
    extra_corner_points_group.append(extra_plane4)
    extra_corner_points_group.append(extra_plane5)
    extra_corner_points_group.append(extra_plane6)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[corner_points_group[35],corner_points_group[26]],
                    [extra_corner_points_group[3],extra_corner_points_group[2]],
                    [extra_corner_points_group[5],extra_corner_points_group[4]],
                    [extra_corner_points_group[0],corner_points_group[30]],
                    [corner_points_group[31],extra_corner_points_group[0]],
                    [corner_points_group[32],corner_points_group[27]],
                    [corner_points_group[33],corner_points_group[28]],
                    [corner_points_group[34],extra_corner_points_group[1]],
                    [extra_corner_points_group[1],corner_points_group[29]]
                   ]

    refer = [get_common_vertices(corner_points_group[0],corner_points_group[1]),
             get_common_vertices(corner_points_group[19],corner_points_group[0]),
             get_common_vertices(corner_points_group[12],corner_points_group[2]),
             get_common_vertices(corner_points_group[2],corner_points_group[3]),
             get_common_vertices(corner_points_group[4],corner_points_group[14]),
             get_common_vertices(corner_points_group[22],corner_points_group[23]),
             get_common_vertices(corner_points_group[22],corner_points_group[23]),
             get_common_vertices(corner_points_group[7],corner_points_group[24]),
             get_common_vertices(corner_points_group[6],corner_points_group[20])
            ]
elif classification_res == 4:
    extra_corner_points_group = []

    extra_plane1 = get_new_plane(corner_points_group[10],corner_points_group[12],corner_points_group[8],corner_points_group[20])
    extra_plane2 = get_new_plane(corner_points_group[4],corner_points_group[13],corner_points_group[2],corner_points_group[20])
    extra_plane3 = get_new_plane(corner_points_group[7],corner_points_group[14],corner_points_group[11],corner_points_group[20])
    extra_plane4 = get_new_plane(corner_points_group[1],corner_points_group[15],corner_points_group[5],corner_points_group[20])

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)
    extra_corner_points_group.append(extra_plane4)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[corner_points_group[21],corner_points_group[12]],
                    [corner_points_group[22],corner_points_group[13]],
                    [corner_points_group[23],corner_points_group[14]],
                    [corner_points_group[24],corner_points_group[15]],

                    [corner_points_group[25],extra_corner_points_group[0]],
                    [corner_points_group[26],extra_corner_points_group[1]],
                    [corner_points_group[27],extra_corner_points_group[2]],
                    [corner_points_group[28],extra_corner_points_group[3]],

                    [extra_corner_points_group[0],corner_points_group[16]],
                    [extra_corner_points_group[1],corner_points_group[17]],
                    [extra_corner_points_group[2],corner_points_group[18]],
                    [extra_corner_points_group[3],corner_points_group[19]],

                    [corner_points_group[29],corner_points_group[20]],
                   ]

    refer = [get_common_vertices(corner_points_group[0],corner_points_group[9]),
             get_common_vertices(corner_points_group[3],corner_points_group[9]),
             get_common_vertices(corner_points_group[3],corner_points_group[6]),
             get_common_vertices(corner_points_group[0],corner_points_group[6]),

             get_common_vertices(get_common_vertices(corner_points_group[1],corner_points_group[10]),corner_points_group[12])+ 
             get_common_vertices(get_common_vertices(corner_points_group[21],corner_points_group[24]),corner_points_group[25]), 

             get_common_vertices(get_common_vertices(corner_points_group[4],corner_points_group[10]),corner_points_group[13])+ 
             get_common_vertices(get_common_vertices(corner_points_group[21],corner_points_group[22]),corner_points_group[26]), 

             get_common_vertices(get_common_vertices(corner_points_group[4],corner_points_group[7]),corner_points_group[14])+ 
             get_common_vertices(get_common_vertices(corner_points_group[22],corner_points_group[23]),corner_points_group[27]), 

             get_common_vertices(get_common_vertices(corner_points_group[1],corner_points_group[7]),corner_points_group[15])+ 
             get_common_vertices(get_common_vertices(corner_points_group[23],corner_points_group[24]),corner_points_group[28]), 

             get_common_vertices(corner_points_group[1],corner_points_group[10]),
             get_common_vertices(corner_points_group[4],corner_points_group[10]),
             get_common_vertices(corner_points_group[4],corner_points_group[7]),
             get_common_vertices(corner_points_group[1],corner_points_group[7]),

             get_common_vertices(get_common_vertices(corner_points_group[5],corner_points_group[8]),corner_points_group[20])+ 
             get_common_vertices(get_common_vertices(corner_points_group[25],corner_points_group[28]),corner_points_group[29]), 
            ]
elif classification_res == 5:
    extra_corner_points_group = []

    extra_plane1 = get_new_plane(corner_points_group[0],corner_points_group[20],corner_points_group[4],corner_points_group[18])
    extra_plane2 = get_new_plane(corner_points_group[13],corner_points_group[14],corner_points_group[11],corner_points_group[21])
    extra_plane3 = get_new_plane(corner_points_group[5],corner_points_group[6],corner_points_group[3],corner_points_group[21])
    extra_plane4 = get_new_plane(corner_points_group[9],corner_points_group[10],corner_points_group[15],corner_points_group[21])
    extra_plane5 = get_new_plane(corner_points_group[1],corner_points_group[2],corner_points_group[7],corner_points_group[21])

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)
    extra_corner_points_group.append(extra_plane4)
    extra_corner_points_group.append(extra_plane5)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[extra_corner_points_group[0],corner_points_group[16]],
                    [extra_corner_points_group[1],corner_points_group[17]],
                    [extra_corner_points_group[2],corner_points_group[18]],
                    [extra_corner_points_group[3],corner_points_group[19]],
                    [extra_corner_points_group[4],corner_points_group[20]],
                    [corner_points_group[21],extra_corner_points_group[0]],
                    [corner_points_group[22],extra_corner_points_group[1]],
                    [corner_points_group[23],extra_corner_points_group[2]],
                    [corner_points_group[24],extra_corner_points_group[3]],
                    [corner_points_group[25],extra_corner_points_group[4]]
                   ]

    refer = [get_common_vertices(corner_points_group[0],corner_points_group[12]),
             get_common_vertices(corner_points_group[5],corner_points_group[13]),
             get_common_vertices(corner_points_group[5],corner_points_group[9]),
             get_common_vertices(corner_points_group[1],corner_points_group[9]),
             get_common_vertices(corner_points_group[1],corner_points_group[13]),
             get_common_vertices(get_common_vertices(corner_points_group[7],corner_points_group[11]),corner_points_group[21])+ 
             get_common_vertices(get_common_vertices(corner_points_group[12],corner_points_group[17]),corner_points_group[20]), 
             get_common_vertices(corner_points_group[2],corner_points_group[14]),
             get_common_vertices(corner_points_group[14],corner_points_group[6]),
             get_common_vertices(corner_points_group[6],corner_points_group[10]),
             get_common_vertices(corner_points_group[10],corner_points_group[2])
            ]
elif classification_res == 6:
    extra_corner_points_group = []

    extra_plane1 = get_new_plane(corner_points_group[6],corner_points_group[8],corner_points_group[10],corner_points_group[12])
    extra_plane2 = get_new_plane(corner_points_group[8],corner_points_group[18],corner_points_group[12],corner_points_group[16])

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[corner_points_group[25],corner_points_group[15]],
                    [extra_corner_points_group[0],corner_points_group[14]],
                    [extra_corner_points_group[1],extra_corner_points_group[0]],
                    [corner_points_group[20],extra_corner_points_group[1]],
                    [corner_points_group[21],corner_points_group[16]],
                    [corner_points_group[22],corner_points_group[17]],
                    [corner_points_group[23],corner_points_group[18]],
                    [corner_points_group[24],corner_points_group[19]]
                   ]

    refer = [get_common_vertices(corner_points_group[0],corner_points_group[11]),
             get_common_vertices(corner_points_group[10],corner_points_group[11]),
             get_common_vertices(corner_points_group[1],corner_points_group[12]),
             get_common_vertices(get_common_vertices(corner_points_group[1],corner_points_group[16]),corner_points_group[19]) + 
             get_common_vertices(get_common_vertices(corner_points_group[20],corner_points_group[21]),corner_points_group[24]),
             get_common_vertices(corner_points_group[13],corner_points_group[2]),
             get_common_vertices(corner_points_group[5],corner_points_group[13]),
             get_common_vertices(corner_points_group[9],corner_points_group[5]),
             get_common_vertices(corner_points_group[2],corner_points_group[9])
            ]
elif classification_res == 7:
    extra_corner_points_group = []
    extra_plane1 = get_new_plane(corner_points_group[0],corner_points_group[19],corner_points_group[4],corner_points_group[5])
    extra_plane2 = get_new_plane(corner_points_group[29],corner_points_group[2],corner_points_group[5],corner_points_group[6])
    extra_plane3 = get_new_plane(corner_points_group[2],corner_points_group[23],corner_points_group[6],corner_points_group[21])

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[extra_corner_points_group[0],corner_points_group[18]],
                    [extra_corner_points_group[1],extra_corner_points_group[0]],
                    [corner_points_group[29],corner_points_group[19]],
                    [extra_corner_points_group[2],extra_corner_points_group[1]],
                    [corner_points_group[24],extra_corner_points_group[2]],
                    [corner_points_group[25],corner_points_group[20]],
                    [corner_points_group[26],corner_points_group[21]],
                    [corner_points_group[27],corner_points_group[22]],
                    [corner_points_group[28],corner_points_group[23]]
                   ]

    refer = [get_common_vertices(corner_points_group[0],corner_points_group[8]),
             get_common_vertices(corner_points_group[9],corner_points_group[10]),
             get_common_vertices(corner_points_group[9],corner_points_group[10]),
             get_common_vertices(corner_points_group[6],corner_points_group[11]),
             get_common_vertices(get_common_vertices(corner_points_group[6],corner_points_group[21]),corner_points_group[22])+ 
             get_common_vertices(get_common_vertices(corner_points_group[24],corner_points_group[26]),corner_points_group[27]),
             get_common_vertices(corner_points_group[3],corner_points_group[17]),
             get_common_vertices(corner_points_group[7],corner_points_group[17]),
             get_common_vertices(corner_points_group[7],corner_points_group[12]),
             get_common_vertices(corner_points_group[3],corner_points_group[17])
            ]
elif classification_res == 8:
    new_vertex_1 = create_inner_vertex(v_co,get_single_vertex(60,59,57),get_single_vertex(23,25,26))
    new_vertex_2 = create_inner_vertex(v_co,get_single_vertex(57,58,59),get_single_vertex(34,36,37))
    new_vertex_3 = create_inner_vertex(v_co,get_single_vertex(46,49,50),get_single_vertex(21,22,24))
    new_vertex_4 = create_inner_vertex(v_co,get_single_vertex(48,50,51),get_single_vertex(32,33,36))

    new_vertex_id_1 = len(v_co) + 0
    new_vertex_id_2 = len(v_co) + 1
    new_vertex_id_3 = len(v_co) + 2
    new_vertex_id_4 = len(v_co) + 3

    v_co = np.vstack((v_co,new_vertex_1))
    v_co = np.vstack((v_co,new_vertex_2))
    v_co = np.vstack((v_co,new_vertex_3))
    v_co = np.vstack((v_co,new_vertex_4))
    
    extra_corner_points_group = []

    extra_plane1 = get_new_plane(corner_points_group[3],corner_points_group[6],corner_points_group[19],corner_points_group[60])
    extra_plane2 = get_new_plane(corner_points_group[5],corner_points_group[7],corner_points_group[18],corner_points_group[58])
    extra_plane3 = get_new_plane(corner_points_group[8],corner_points_group[60],corner_points_group[13],corner_points_group[16])
    extra_plane4 = get_new_plane(corner_points_group[9],corner_points_group[58],corner_points_group[15],corner_points_group[17])
    extra_plane5 = get_common_vertices(corner_points_group[1],corner_points_group[3]) + get_single_vertex(23,25,26) + [new_vertex_id_1]
    extra_plane6 = get_common_vertices(corner_points_group[52],corner_points_group[4]) + [new_vertex_id_1] + [new_vertex_id_2]
    extra_plane7 = get_common_vertices(corner_points_group[2],corner_points_group[5]) + get_single_vertex(34,36,37) + [new_vertex_id_2]
    extra_plane8 = get_common_vertices(corner_points_group[22],corner_points_group[25]) + [new_vertex_id_1] + [new_vertex_id_3]
    extra_plane9 = [new_vertex_id_1] + [new_vertex_id_2] + [new_vertex_id_3] + [new_vertex_id_4]
    extra_plane10 = get_common_vertices(corner_points_group[33],corner_points_group[36]) + [new_vertex_id_2] + [new_vertex_id_4]
    extra_plane11 = get_common_vertices(corner_points_group[21],corner_points_group[24]) + get_single_vertex(10,11,13) + [new_vertex_id_3]
    extra_plane12 = get_common_vertices(corner_points_group[11],corner_points_group[14]) + [new_vertex_id_3] + [new_vertex_id_4]
    extra_plane13 = get_common_vertices(corner_points_group[32],corner_points_group[35]) + get_single_vertex(11,12,15) + [new_vertex_id_4]

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)
    extra_corner_points_group.append(extra_plane4)
    extra_corner_points_group.append(extra_plane5)
    extra_corner_points_group.append(extra_plane6)
    extra_corner_points_group.append(extra_plane7)
    extra_corner_points_group.append(extra_plane8)
    extra_corner_points_group.append(extra_plane9)
    extra_corner_points_group.append(extra_plane10)
    extra_corner_points_group.append(extra_plane11)
    extra_corner_points_group.append(extra_plane12)
    extra_corner_points_group.append(extra_plane13)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[corner_points_group[52],corner_points_group[42]],
                    [corner_points_group[54],extra_corner_points_group[0]],
                    [corner_points_group[53],extra_corner_points_group[1]],
                    [corner_points_group[56],extra_corner_points_group[2]],
                    [corner_points_group[55],extra_corner_points_group[3]],
                    [extra_corner_points_group[0],extra_corner_points_group[4]],
                    [corner_points_group[57],extra_corner_points_group[5]],
                    [extra_corner_points_group[1],extra_corner_points_group[6]],
                    [corner_points_group[60],extra_corner_points_group[7]],
                    [corner_points_group[59],extra_corner_points_group[8]],
                    [corner_points_group[58],extra_corner_points_group[9]],
                    [extra_corner_points_group[2],extra_corner_points_group[10]],
                    [corner_points_group[61],extra_corner_points_group[11]],
                    [extra_corner_points_group[3],extra_corner_points_group[12]],
                    [extra_corner_points_group[4],corner_points_group[43]],
                    [extra_corner_points_group[5],corner_points_group[44]],
                    [extra_corner_points_group[6],corner_points_group[45]],
                    [extra_corner_points_group[7],corner_points_group[46]],
                    [extra_corner_points_group[8],corner_points_group[47]],
                    [extra_corner_points_group[9],corner_points_group[48]],
                    [extra_corner_points_group[10],corner_points_group[49]],
                    [extra_corner_points_group[11],corner_points_group[50]],
                    [extra_corner_points_group[12],corner_points_group[51]],
                   ]

    refer = [get_common_vertices(corner_points_group[0],corner_points_group[20]),
             get_common_vertices(corner_points_group[6],corner_points_group[28]),
             get_common_vertices(corner_points_group[7],corner_points_group[30]),
             get_common_vertices(corner_points_group[8],corner_points_group[27]),
             get_common_vertices(corner_points_group[9],corner_points_group[29]),
             get_common_vertices(corner_points_group[3],corner_points_group[26]),
             get_common_vertices(corner_points_group[3],corner_points_group[4]),
             get_common_vertices(corner_points_group[4],corner_points_group[5]),
             get_common_vertices(corner_points_group[24],corner_points_group[25]),
             get_single_vertex(19,59,60) + [new_vertex_id_1],
             get_common_vertices(corner_points_group[36],corner_points_group[37]),
             get_common_vertices(corner_points_group[24],corner_points_group[25]),
             get_common_vertices(corner_points_group[13],corner_points_group[14]),
             get_common_vertices(corner_points_group[14],corner_points_group[15]),
             get_common_vertices(corner_points_group[22],corner_points_group[23]),
             get_common_vertices(corner_points_group[1],corner_points_group[20]),
             get_common_vertices(corner_points_group[31],corner_points_group[2]),
             get_common_vertices(corner_points_group[22],corner_points_group[23]),
             get_single_vertex(46,47,44) + [new_vertex_id_1],
             get_common_vertices(corner_points_group[32],corner_points_group[33]),
             get_common_vertices(corner_points_group[21],corner_points_group[22]),
             get_common_vertices(corner_points_group[10],corner_points_group[11]),
             get_common_vertices(corner_points_group[11],corner_points_group[12])
            ]
elif classification_res == 9:
    new_vertex_1 = create_inner_vertex(v_co,get_single_vertex(47,48,44),get_single_vertex(24,25,26))
    new_vertex_2 = create_inner_vertex(v_co,get_single_vertex(42,46,47),get_single_vertex(24,25,26))
    new_vertex_3 = create_inner_vertex(v_co,get_single_vertex(41,42,43),get_single_vertex(35,36,37))

    new_vertex_id_1 = len(v_co) + 0
    new_vertex_id_2 = len(v_co) + 1
    new_vertex_id_3 = len(v_co) + 2

    v_co = np.vstack((v_co,new_vertex_1))
    v_co = np.vstack((v_co,new_vertex_2))
    v_co = np.vstack((v_co,new_vertex_3))
    
    extra_corner_points_group = []

    extra_plane1 = get_new_plane(corner_points_group[12],corner_points_group[13],corner_points_group[14],corner_points_group[15])
    extra_plane2 = get_common_vertices(corner_points_group[25],corner_points_group[26]) + get_single_vertex(6,7,8) + [new_vertex_id_1]
    extra_plane3 = get_common_vertices(corner_points_group[24],corner_points_group[25]) + get_single_vertex(22,23,11) + [new_vertex_id_1]
    extra_plane4 = get_common_vertices(corner_points_group[16],corner_points_group[17]) + get_single_vertex(24,25,26) + [new_vertex_id_1]
    extra_plane5 = get_new_plane(corner_points_group[23],corner_points_group[50],corner_points_group[32],corner_points_group[52])
    extra_plane6 = get_new_plane(corner_points_group[22],corner_points_group[23],corner_points_group[31],corner_points_group[32])
    extra_plane7 = get_common_vertices(corner_points_group[5],corner_points_group[6]) + [new_vertex_id_1] + [new_vertex_id_2]
    extra_plane8 =  get_single_vertex(22,23,11) + get_single_vertex(31,32,9) + [new_vertex_id_1] + [new_vertex_id_2]
    extra_plane9 =  get_single_vertex(16,17,29)  + [new_vertex_id_1] + [new_vertex_id_2] + [new_vertex_id_3]
    extra_plane10 = get_common_vertices(corner_points_group[18],corner_points_group[19]) + get_single_vertex(16,17,29) + [new_vertex_id_3]    
    extra_plane11 = get_common_vertices(corner_points_group[2],corner_points_group[3]) + get_single_vertex(33,34,35) + [new_vertex_id_2]      
    extra_plane12 = get_common_vertices(corner_points_group[9],corner_points_group[52]) + get_single_vertex(33,34,35) + [new_vertex_id_2]          
    extra_plane13 = get_common_vertices(corner_points_group[35],corner_points_group[53]) + [new_vertex_id_2] + [new_vertex_id_3]
    extra_plane14 = get_common_vertices(corner_points_group[36],corner_points_group[37]) + get_single_vertex(19,20,21) + [new_vertex_id_3]  

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)
    extra_corner_points_group.append(extra_plane4)
    extra_corner_points_group.append(extra_plane5)
    extra_corner_points_group.append(extra_plane6)
    extra_corner_points_group.append(extra_plane7)
    extra_corner_points_group.append(extra_plane8)
    extra_corner_points_group.append(extra_plane9)
    extra_corner_points_group.append(extra_plane10)
    extra_corner_points_group.append(extra_plane11)
    extra_corner_points_group.append(extra_plane12)
    extra_corner_points_group.append(extra_plane13)
    extra_corner_points_group.append(extra_plane14)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')
    
    plane_groups = [[extra_corner_points_group[0],corner_points_group[45]],
                    [corner_points_group[61],extra_corner_points_group[0]],
                    [corner_points_group[62],extra_corner_points_group[1]],
                    [extra_corner_points_group[1],corner_points_group[50]],
                    [extra_corner_points_group[2],corner_points_group[48]],
                    [corner_points_group[60],extra_corner_points_group[3]],
                    [extra_corner_points_group[3],corner_points_group[44]],
                    [corner_points_group[63],extra_corner_points_group[6]],
                    [extra_corner_points_group[6],extra_corner_points_group[4]],
                    [extra_corner_points_group[4],corner_points_group[51]],
                    [extra_corner_points_group[5],corner_points_group[49]],
                    [extra_corner_points_group[7],corner_points_group[47]],
                    [corner_points_group[59],extra_corner_points_group[8]],
                    [extra_corner_points_group[8],corner_points_group[43]],
                    [corner_points_group[57],extra_corner_points_group[9]],
                    [extra_corner_points_group[9],corner_points_group[41]],
                    [corner_points_group[64],extra_corner_points_group[10]],
                    [extra_corner_points_group[10],corner_points_group[52]],
                    [extra_corner_points_group[11],corner_points_group[46]],
                    [corner_points_group[58],extra_corner_points_group[12]],
                    [extra_corner_points_group[12],corner_points_group[42]],
                    [corner_points_group[56],extra_corner_points_group[13]],
                    [extra_corner_points_group[13],corner_points_group[40]],
                    [corner_points_group[66],corner_points_group[54]],
                    [corner_points_group[65],corner_points_group[53]],
                    [corner_points_group[67],corner_points_group[55]]
                   ]

    refer = [get_common_vertices(corner_points_group[12],corner_points_group[27]),
             get_common_vertices(corner_points_group[13],corner_points_group[28]),
             get_common_vertices(corner_points_group[8],corner_points_group[26]),
             get_common_vertices(corner_points_group[7],corner_points_group[25]),
             get_common_vertices(corner_points_group[11],corner_points_group[24]),
             get_common_vertices(corner_points_group[15],corner_points_group[17]),
             get_common_vertices(corner_points_group[14],corner_points_group[16]),
             get_common_vertices(corner_points_group[6],corner_points_group[8]),
             get_common_vertices(corner_points_group[5],corner_points_group[7]),
             get_common_vertices(corner_points_group[4],corner_points_group[23]),
             get_common_vertices(corner_points_group[10],corner_points_group[22]),
             get_common_vertices(corner_points_group[22],corner_points_group[11]),
             get_common_vertices(corner_points_group[17],corner_points_group[30]),
             get_common_vertices(corner_points_group[16],corner_points_group[29]),
             get_common_vertices(corner_points_group[19],corner_points_group[30]),
             get_common_vertices(corner_points_group[18],corner_points_group[29]),
             get_common_vertices(corner_points_group[1],corner_points_group[3]),
             get_common_vertices(corner_points_group[2],corner_points_group[34]),
             get_common_vertices(corner_points_group[9],corner_points_group[33]),
             get_common_vertices(corner_points_group[37],corner_points_group[38]),
             get_common_vertices(corner_points_group[35],corner_points_group[36]),
             get_common_vertices(corner_points_group[37],corner_points_group[38]),
             get_common_vertices(corner_points_group[35],corner_points_group[36]),
             get_common_vertices(corner_points_group[0],corner_points_group[1]),
             get_common_vertices(corner_points_group[37],corner_points_group[38]),
             get_common_vertices(corner_points_group[0],corner_points_group[1])
            ]
elif classification_res == 10:
    new_vertex_1 = create_inner_vertex_with_lines_and_planes(83,84,85,82,83,85,84,86,19,17,99,47,15,97,55)
    new_vertex_2 = create_inner_vertex_with_lines_and_planes(83,84,85,82,83,85,84,86,19,99,100,48,97,98,55)
    new_vertex_3 = create_inner_vertex_with_lines_and_planes(83,84,85,82,83,85,84,86,19,100,18,48,22,98,56)
    new_vertex_4 = create_inner_vertex_with_lines_and_planes(83,84,85,82,83,85,84,86,19,18,125,48,22,123,56)
    new_vertex_5 = create_inner_vertex_with_lines_and_planes(83,84,85,82,83,85,84,86,19,124,125,48,122,123,55)
    new_vertex_6 = create_inner_vertex_with_lines_and_planes(83,84,85,82,83,85,84,86,19,17,124,47,15,122,55)

    new_vertex_7 = create_inner_vertex_with_lines_and_planes(85,86,87,82,85,43,107,110,46,17,99,47,15,97,55)
    new_vertex_8 = create_inner_vertex_with_lines_and_planes(85,86,87,82,85,43,107,110,46,99,100,48,97,98,55)
    new_vertex_9 = create_inner_vertex_with_lines_and_planes(85,86,87,82,85,43,107,110,46,100,18,48,22,98,56)
    new_vertex_10 = create_inner_vertex_with_lines_and_planes(85,86,87,82,85,43,107,110,46,18,125,48,22,123,56)
    new_vertex_11 = create_inner_vertex_with_lines_and_planes(85,86,87,82,85,43,107,110,46,124,125,48,122,123,55)
    new_vertex_12 = create_inner_vertex_with_lines_and_planes(85,86,87,82,85,43,107,110,46,17,124,47,15,122,55)

    new_vertex_1 = np.around(new_vertex_1,3)
    new_vertex_2 = np.around(new_vertex_2,3)
    new_vertex_3 = np.around(new_vertex_3,3)
    new_vertex_4 = np.around(new_vertex_4,3)
    new_vertex_5 = np.around(new_vertex_5,3)
    new_vertex_6 = np.around(new_vertex_6,3)
    new_vertex_7 = np.around(new_vertex_7,3)
    new_vertex_8 = np.around(new_vertex_8,3)
    new_vertex_9 = np.around(new_vertex_9,3)
    new_vertex_10 = np.around(new_vertex_10,3)
    new_vertex_11 = np.around(new_vertex_11,3)
    new_vertex_12 = np.around(new_vertex_12,3)

    new_vertex_id_1 = len(v_co) + 0
    new_vertex_id_2 = len(v_co) + 1
    new_vertex_id_3 = len(v_co) + 2
    new_vertex_id_4 = len(v_co) + 3
    new_vertex_id_5 = len(v_co) + 4
    new_vertex_id_6 = len(v_co) + 5
    new_vertex_id_7 = len(v_co) + 6
    new_vertex_id_8 = len(v_co) + 7
    new_vertex_id_9 = len(v_co) + 8
    new_vertex_id_10 = len(v_co) + 9
    new_vertex_id_11 = len(v_co) + 10
    new_vertex_id_12 = len(v_co) + 11

    v_co = np.vstack((v_co,new_vertex_1))
    v_co = np.vstack((v_co,new_vertex_2))
    v_co = np.vstack((v_co,new_vertex_3))
    v_co = np.vstack((v_co,new_vertex_4))
    v_co = np.vstack((v_co,new_vertex_5))
    v_co = np.vstack((v_co,new_vertex_6))
    v_co = np.vstack((v_co,new_vertex_7))
    v_co = np.vstack((v_co,new_vertex_8))
    v_co = np.vstack((v_co,new_vertex_9))
    v_co = np.vstack((v_co,new_vertex_10))
    v_co = np.vstack((v_co,new_vertex_11))
    v_co = np.vstack((v_co,new_vertex_12))

    extra_corner_points_group = []
    
    extra_plane1 = get_common_vertices(corner_points_group[0],corner_points_group[2]) + get_single_vertex(70,71,72) + get_single_vertex(5,102,103)
    extra_plane2 = get_common_vertices(corner_points_group[32],corner_points_group[34]) + get_single_vertex(70,71,72) + get_single_vertex(5,102,103)
    extra_plane3 = get_new_plane(corner_points_group[3],corner_points_group[77],corner_points_group[30],corner_points_group[32])
    extra_plane4 = get_new_plane(corner_points_group[5],corner_points_group[103],corner_points_group[31],corner_points_group[33])
    extra_plane5 = get_new_plane(corner_points_group[4],corner_points_group[78],corner_points_group[29],corner_points_group[31])
    extra_plane6 = get_new_plane(corner_points_group[38],corner_points_group[104],corner_points_group[75],corner_points_group[76])
    extra_plane7 = get_new_plane(corner_points_group[35],corner_points_group[79],corner_points_group[74],corner_points_group[75])
    extra_plane8 = get_new_plane(corner_points_group[25],corner_points_group[105],corner_points_group[11],corner_points_group[14])
    extra_plane9 = get_new_plane(corner_points_group[24],corner_points_group[80],corner_points_group[8],corner_points_group[11])
    extra_plane10 = get_common_vertices(corner_points_group[10],corner_points_group[13]) + get_single_vertex(67,68,28) + get_single_vertex(25,45,105)
    extra_plane11 = get_common_vertices(corner_points_group[7],corner_points_group[10]) + get_single_vertex(66,67,26) + get_single_vertex(24,42,80)
    extra_plane12 = get_new_plane(corner_points_group[9],corner_points_group[12],corner_points_group[27],corner_points_group[28])
    extra_plane13 = get_new_plane(corner_points_group[6],corner_points_group[9],corner_points_group[26],corner_points_group[27])
    extra_plane14 = get_new_plane(corner_points_group[40],corner_points_group[41],corner_points_group[70],corner_points_group[71])
    extra_plane15 = get_new_plane(corner_points_group[39],corner_points_group[40],corner_points_group[69],corner_points_group[70])
    extra_plane16 = get_new_plane(corner_points_group[45],corner_points_group[106],corner_points_group[67],corner_points_group[68])
    extra_plane17 = get_new_plane(corner_points_group[42],corner_points_group[81],corner_points_group[66],corner_points_group[67])
    extra_plane18 = get_new_plane(corner_points_group[46],corner_points_group[107],corner_points_group[64],corner_points_group[65])
    extra_plane19 = get_new_plane(corner_points_group[43],corner_points_group[82],corner_points_group[63],corner_points_group[64])

    extra_plane20 = get_common_vertices(corner_points_group[49],corner_points_group[54]) + get_single_vertex(16,108,110) + [new_vertex_id_6]
    extra_plane21 = get_common_vertices(corner_points_group[49],corner_points_group[50]) + get_single_vertex(16,83,85) + [new_vertex_id_1]
    extra_plane22 = get_common_vertices(corner_points_group[54],corner_points_group[124]) + [new_vertex_id_5] + [new_vertex_id_6]
    extra_plane23 = get_common_vertices(corner_points_group[50],corner_points_group[99]) + [new_vertex_id_1] + [new_vertex_id_2]
    extra_plane24 = get_common_vertices(corner_points_group[53],corner_points_group[125]) + [new_vertex_id_4] + [new_vertex_id_5]
    extra_plane25 = get_common_vertices(corner_points_group[51],corner_points_group[100]) + [new_vertex_id_2] + [new_vertex_id_3]
    extra_plane26 = get_common_vertices(corner_points_group[52],corner_points_group[53]) + get_single_vertex(19,20,111) + [new_vertex_id_4]
    extra_plane27 = get_common_vertices(corner_points_group[51],corner_points_group[52]) + get_single_vertex(84,86,20) + [new_vertex_id_3]

    extra_plane28 = get_common_vertices(corner_points_group[107],corner_points_group[110]) + [new_vertex_id_6] + [new_vertex_id_12]
    extra_plane29 = get_common_vertices(corner_points_group[82],corner_points_group[85]) + [new_vertex_id_1] + [new_vertex_id_7]
    extra_plane30 = [new_vertex_id_5] + [new_vertex_id_6] + [new_vertex_id_11] + [new_vertex_id_12]
    extra_plane31 = [new_vertex_id_1] + [new_vertex_id_2] + [new_vertex_id_7] + [new_vertex_id_8]
    extra_plane32 = [new_vertex_id_4] + [new_vertex_id_5] + [new_vertex_id_10] + [new_vertex_id_11]
    extra_plane33 = [new_vertex_id_2] + [new_vertex_id_3] + [new_vertex_id_8] + [new_vertex_id_9]
    extra_plane34 = get_common_vertices(corner_points_group[20],corner_points_group[111]) + [new_vertex_id_4] + [new_vertex_id_10]
    extra_plane35 = get_common_vertices(corner_points_group[20],corner_points_group[86]) + [new_vertex_id_3] + [new_vertex_id_9]

    extra_plane36 = get_common_vertices(corner_points_group[57],corner_points_group[62]) + get_single_vertex(46,107,110) + [new_vertex_id_12]
    extra_plane37 = get_common_vertices(corner_points_group[57],corner_points_group[58]) + get_single_vertex(43,82,85) + [new_vertex_id_7]
    extra_plane38 = get_common_vertices(corner_points_group[62],corner_points_group[122]) + [new_vertex_id_11] + [new_vertex_id_12]
    extra_plane39 = get_common_vertices(corner_points_group[58],corner_points_group[97]) + [new_vertex_id_7] + [new_vertex_id_8]
    extra_plane40 = get_common_vertices(corner_points_group[61],corner_points_group[123]) + [new_vertex_id_10] + [new_vertex_id_11]
    extra_plane41 = get_common_vertices(corner_points_group[59],corner_points_group[98]) + [new_vertex_id_8] + [new_vertex_id_9]
    extra_plane42 = get_common_vertices(corner_points_group[60],corner_points_group[61]) + get_single_vertex(20,21,112) + [new_vertex_id_10]
    extra_plane43 = get_common_vertices(corner_points_group[59],corner_points_group[60]) + get_single_vertex(20,21,87) + [new_vertex_id_9]

    extra_plane44 = get_new_plane(corner_points_group[110],corner_points_group[46],corner_points_group[62],corner_points_group[65])
    extra_plane45 = get_new_plane(corner_points_group[58],corner_points_group[63],corner_points_group[43],corner_points_group[85])

    extra_corner_points_group.append(extra_plane1)
    extra_corner_points_group.append(extra_plane2)
    extra_corner_points_group.append(extra_plane3)
    extra_corner_points_group.append(extra_plane4)
    extra_corner_points_group.append(extra_plane5)
    extra_corner_points_group.append(extra_plane6)
    extra_corner_points_group.append(extra_plane7)
    extra_corner_points_group.append(extra_plane8)
    extra_corner_points_group.append(extra_plane9)
    extra_corner_points_group.append(extra_plane10)
    extra_corner_points_group.append(extra_plane11)
    extra_corner_points_group.append(extra_plane12)
    extra_corner_points_group.append(extra_plane13)
    extra_corner_points_group.append(extra_plane14)
    extra_corner_points_group.append(extra_plane15)
    extra_corner_points_group.append(extra_plane16)
    extra_corner_points_group.append(extra_plane17)
    extra_corner_points_group.append(extra_plane18)
    extra_corner_points_group.append(extra_plane19)
    extra_corner_points_group.append(extra_plane20)
    extra_corner_points_group.append(extra_plane21)
    extra_corner_points_group.append(extra_plane22)
    extra_corner_points_group.append(extra_plane23)
    extra_corner_points_group.append(extra_plane24)
    extra_corner_points_group.append(extra_plane25)
    extra_corner_points_group.append(extra_plane26)
    extra_corner_points_group.append(extra_plane27)
    extra_corner_points_group.append(extra_plane28)
    extra_corner_points_group.append(extra_plane29)
    extra_corner_points_group.append(extra_plane30)
    extra_corner_points_group.append(extra_plane31)
    extra_corner_points_group.append(extra_plane32)
    extra_corner_points_group.append(extra_plane33)
    extra_corner_points_group.append(extra_plane34)
    extra_corner_points_group.append(extra_plane35)
    extra_corner_points_group.append(extra_plane36)
    extra_corner_points_group.append(extra_plane37)
    extra_corner_points_group.append(extra_plane38)
    extra_corner_points_group.append(extra_plane39)
    extra_corner_points_group.append(extra_plane40)
    extra_corner_points_group.append(extra_plane41)
    extra_corner_points_group.append(extra_plane42)
    extra_corner_points_group.append(extra_plane43)
    extra_corner_points_group.append(extra_plane44)
    extra_corner_points_group.append(extra_plane45)

    extra_corner_points_group = np.array(extra_corner_points_group).astype('int')

    plane_groups = [[corner_points_group[101],extra_corner_points_group[0]],
                    [extra_corner_points_group[0],corner_points_group[77]],
                    [corner_points_group[118],extra_corner_points_group[1]],
                    [extra_corner_points_group[1],extra_corner_points_group[2]],
                    [extra_corner_points_group[2],corner_points_group[93]],
                    [corner_points_group[103],corner_points_group[78]],
                    [corner_points_group[119],extra_corner_points_group[3]],
                    [extra_corner_points_group[3],extra_corner_points_group[4]],
                    [extra_corner_points_group[4],corner_points_group[94]],
                    [corner_points_group[104],corner_points_group[79]],
                    [corner_points_group[120],extra_corner_points_group[5]],
                    [extra_corner_points_group[5],extra_corner_points_group[6]],
                    [extra_corner_points_group[6],corner_points_group[95]],
                    [corner_points_group[105],corner_points_group[80]],
                    [corner_points_group[121],extra_corner_points_group[7]],
                    [extra_corner_points_group[7],extra_corner_points_group[8]],
                    [extra_corner_points_group[8],corner_points_group[96]],
                    [corner_points_group[115],extra_corner_points_group[9]],
                    [extra_corner_points_group[9],extra_corner_points_group[10]],
                    [extra_corner_points_group[10],corner_points_group[90]],
                    [corner_points_group[116],extra_corner_points_group[11]],
                    [extra_corner_points_group[11],extra_corner_points_group[12]],
                    [extra_corner_points_group[12],corner_points_group[91]],
                    [corner_points_group[117],extra_corner_points_group[13]],
                    [extra_corner_points_group[13],extra_corner_points_group[14]],
                    [extra_corner_points_group[14],corner_points_group[92]],
                    [corner_points_group[106],corner_points_group[81]],
                    [corner_points_group[114],extra_corner_points_group[15]],
                    [extra_corner_points_group[15],extra_corner_points_group[16]],
                    [extra_corner_points_group[16],corner_points_group[89]],
                    [corner_points_group[107],corner_points_group[82]],
                    [corner_points_group[113],extra_corner_points_group[17]],
                    [extra_corner_points_group[17],extra_corner_points_group[18]],
                    [extra_corner_points_group[18],corner_points_group[88]],

                    [corner_points_group[124],corner_points_group[99]],
                    [corner_points_group[125],corner_points_group[100]],

                    [extra_corner_points_group[19],extra_corner_points_group[20]],
                    [corner_points_group[108],extra_corner_points_group[21]],
                    [extra_corner_points_group[21],extra_corner_points_group[22]],
                    [extra_corner_points_group[22],corner_points_group[83]],
                    [corner_points_group[109],extra_corner_points_group[23]],
                    [extra_corner_points_group[23],extra_corner_points_group[24]],
                    [extra_corner_points_group[24],corner_points_group[84]],
                    [extra_corner_points_group[25],extra_corner_points_group[26]],

                    [extra_corner_points_group[27],extra_corner_points_group[28]],
                    [corner_points_group[110],extra_corner_points_group[29]],
                    [extra_corner_points_group[29],extra_corner_points_group[30]],
                    [extra_corner_points_group[30],corner_points_group[85]],
                    [corner_points_group[111],extra_corner_points_group[31]],
                    [extra_corner_points_group[31],extra_corner_points_group[32]],
                    [extra_corner_points_group[32],corner_points_group[86]],
                    [extra_corner_points_group[33],extra_corner_points_group[34]],

                    [extra_corner_points_group[35],extra_corner_points_group[36]],
                    [extra_corner_points_group[43],extra_corner_points_group[37]],
                    [extra_corner_points_group[37],extra_corner_points_group[38]],
                    [extra_corner_points_group[38],extra_corner_points_group[44]],
                    [corner_points_group[112],extra_corner_points_group[39]],
                    [extra_corner_points_group[39],extra_corner_points_group[40]],
                    [extra_corner_points_group[40],corner_points_group[87]],
                    [extra_corner_points_group[41],extra_corner_points_group[42]],

                    [corner_points_group[122],corner_points_group[97]],
                    [corner_points_group[123],corner_points_group[98]],
                   ]

    refer = [get_common_vertices(corner_points_group[2],corner_points_group[102]),
             get_common_vertices(corner_points_group[0],corner_points_group[1]),
             get_common_vertices(corner_points_group[5],corner_points_group[102]),
             get_common_vertices(corner_points_group[70],corner_points_group[72]),
             get_common_vertices(corner_points_group[3],corner_points_group[4]),
             get_common_vertices(corner_points_group[0],corner_points_group[1]),
             get_common_vertices(corner_points_group[5],corner_points_group[38]),
             get_common_vertices(corner_points_group[31],corner_points_group[32]),
             get_common_vertices(corner_points_group[3],corner_points_group[4]),
             get_common_vertices(corner_points_group[36],corner_points_group[37]),
             get_common_vertices(corner_points_group[38],corner_points_group[25]),
             get_common_vertices(corner_points_group[11],corner_points_group[75]),
             get_common_vertices(corner_points_group[35],corner_points_group[24]),
             get_common_vertices(corner_points_group[36],corner_points_group[37]),
             get_common_vertices(corner_points_group[38],corner_points_group[25]),
             get_common_vertices(corner_points_group[10],corner_points_group[11]),
             get_common_vertices(corner_points_group[35],corner_points_group[24]),
             get_common_vertices(corner_points_group[13],corner_points_group[14]),
             get_common_vertices(corner_points_group[10],corner_points_group[11]),
             get_common_vertices(corner_points_group[7],corner_points_group[8]),
             get_common_vertices(corner_points_group[12],corner_points_group[13]),
             get_common_vertices(corner_points_group[9],corner_points_group[10]),
             get_common_vertices(corner_points_group[6],corner_points_group[7]),
             get_common_vertices(corner_points_group[71],corner_points_group[73]),
             get_common_vertices(corner_points_group[70],corner_points_group[72]),
             get_common_vertices(corner_points_group[69],corner_points_group[3]),
             get_common_vertices(corner_points_group[37],corner_points_group[23]),
             get_common_vertices(corner_points_group[65],corner_points_group[68]),
             get_common_vertices(corner_points_group[64],corner_points_group[67]),
             get_common_vertices(corner_points_group[63],corner_points_group[66]),
             get_common_vertices(corner_points_group[23],corner_points_group[44]),
             get_common_vertices(corner_points_group[65],corner_points_group[68]),
             get_common_vertices(corner_points_group[64],corner_points_group[67]),
             get_common_vertices(corner_points_group[63],corner_points_group[66]), 

             get_common_vertices(corner_points_group[47],corner_points_group[48]),
             get_common_vertices(corner_points_group[47],corner_points_group[48]),

             get_common_vertices(corner_points_group[16],corner_points_group[49]),
             get_common_vertices(corner_points_group[53],corner_points_group[54]),
             get_common_vertices(corner_points_group[49],corner_points_group[17]),
             get_common_vertices(corner_points_group[50],corner_points_group[51]),
             get_common_vertices(corner_points_group[53],corner_points_group[54]),
             get_common_vertices(corner_points_group[18],corner_points_group[52]),
             get_common_vertices(corner_points_group[50],corner_points_group[51]),
             get_common_vertices(corner_points_group[18],corner_points_group[52]),

             [new_vertex_id_7] + [new_vertex_id_12],
             get_single_vertex(110,111,112) + [new_vertex_id_11],
             [new_vertex_id_8] + [new_vertex_id_11],
             get_single_vertex(85,86,87) + [new_vertex_id_8],
             get_single_vertex(110,111,112) + [new_vertex_id_11],
             [new_vertex_id_8] + [new_vertex_id_11],
             get_single_vertex(85,86,87) + [new_vertex_id_8],         
             [new_vertex_id_9] + [new_vertex_id_10],

             [new_vertex_id_7] + [new_vertex_id_12],
             get_single_vertex(110,111,112) + [new_vertex_id_11],
             [new_vertex_id_8] + [new_vertex_id_11],
             get_single_vertex(85,86,87) + [new_vertex_id_8],
             get_single_vertex(110,111,112) + [new_vertex_id_11],
             [new_vertex_id_8] + [new_vertex_id_11],
             get_single_vertex(85,86,87) + [new_vertex_id_8],         
             [new_vertex_id_9] + [new_vertex_id_10],

             get_common_vertices(corner_points_group[55],corner_points_group[56]),
             get_common_vertices(corner_points_group[55],corner_points_group[56]),
            ]


# In[36]:


"""

"""

#
def sortVertsByCCW(p,v_co):
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
    a1 = get_angle(v1[:2],v2[:2])
    a2 = get_angle(v1[:2],v3[:2])
    a3 = get_angle(v2[:2],v3[:2])
    if np.max([a1,a2,a3]) == a1:
        res =  [p[0],p[1],p[3],p[2]]
    elif np.max([a1,a2,a3]) == a2:
        res =  [p[0],p[1],p[2],p[3]]
    else:
        res =  [p[0],p[2],p[1],p[3]] 
    normal1 = (v_co[res[1]] - v_co[res[0]])[:2]
    normal2 = (v_co[res[2]] - v_co[res[0]])[:2]
    if np.cross(normal1,normal2) < 0:
        return res[::-1]
    else:
        return res

def rotate_array(p):
    new_p = []
    for i in range(len(p)):
        new_p.append(p[(i + 1) % len(p)])
    return new_p
    
#
def sortVertsByRHSR(p1,p2,refer):
    p1_r = sortVertsByCCW(p1,v_co)
    while p1_r[0] not in refer:
        p1_r = rotate_array(p1_r)
    p2_r = sortVertsByCCW(p2,v_co)
    while p2_r[0] not in refer:
        p2_r = rotate_array(p2_r)
    res = np.hstack((p1_r,p2_r))
    return res

res_list = []
for i in range(len(plane_groups)):
    p1 = plane_groups[i][0]
    p2 = plane_groups[i][1]
    tmp = sortVertsByRHSR(p1,p2,refer[i])
    res_list.append(list(tmp))


# In[37]:


original_indices = np.sort(np.unique(corner_points_group)).astype('int')

if classification_res == 8:
    original_indices = np.append(original_indices,new_vertex_id_1)
    original_indices = np.append(original_indices,new_vertex_id_2)
    original_indices = np.append(original_indices,new_vertex_id_3)
    original_indices = np.append(original_indices,new_vertex_id_4)
elif classification_res == 9:
    original_indices = np.append(original_indices,new_vertex_id_1)
    original_indices = np.append(original_indices,new_vertex_id_2)
    original_indices = np.append(original_indices,new_vertex_id_3)
elif classification_res == 10:
    original_indices = np.append(original_indices,new_vertex_id_1)
    original_indices = np.append(original_indices,new_vertex_id_2)
    original_indices = np.append(original_indices,new_vertex_id_3)
    original_indices = np.append(original_indices,new_vertex_id_4)
    original_indices = np.append(original_indices,new_vertex_id_5)
    original_indices = np.append(original_indices,new_vertex_id_6)
    original_indices = np.append(original_indices,new_vertex_id_7)
    original_indices = np.append(original_indices,new_vertex_id_8)
    original_indices = np.append(original_indices,new_vertex_id_9)
    original_indices = np.append(original_indices,new_vertex_id_10)
    original_indices = np.append(original_indices,new_vertex_id_11)
    original_indices = np.append(original_indices,new_vertex_id_12)
    
new_indices = np.arange(1,len(original_indices) + 1).astype('int')
res_vertices = v_co[original_indices]
d = dict(zip(original_indices,new_indices))
res_list =  np.vectorize(d.get)(res_list)
res_list = np.array(res_list)
print(res_list)


"""

"""
solid_info = np.zeros((len(res_list),10)).astype('int')
solid_info[:,0] = np.arange(1,len(res_list) + 1)
solid_info[:,1] = 1
solid_info[:,2:] = res_list

solid_info = solid_info.astype('str')

for i in range(solid_info.shape[0]):
    for j in range(solid_info.shape[1]):
        solid_info[i][j] = solid_info[i][j].rjust(8)

np.savetxt(text_file_dir + '/solid_info.txt',solid_info,'%s',delimiter='')        
        
p1 = (np.arange(1,len(res_vertices) + 1)).astype('int').reshape(-1,1).astype('str')
p2 = res_vertices.astype('str')
p3 = np.zeros((len(res_vertices),2)).astype('int').astype('str')

vertex_info = np.hstack((np.hstack((p1,p2)),p3))

for i in range(vertex_info.shape[0]):
    for j in range(vertex_info.shape[1]):
        vertex_info[i][j] = vertex_info[i][j].rjust(8) if (j == 0 or j == 4 or j == 5) else vertex_info[i][j].rjust(16)
np.savetxt(text_file_dir + '/v_vertices_info.txt',vertex_info,fmt = '%s',delimiter = '')    

"""

"""
with open(text_file_dir + '/v_vertices_info.txt','r') as v_info:
    v_i = v_info.readlines()
with open(text_file_dir + '/solid_info.txt','r') as s_info:
    s_i = s_info.readlines()
    
pattern1 = ['$# LS-DYNA Keyword file created by LS-PrePost(R) V4.5.3 - 28Oct2017',
'$# Created on Mar-15-2018 (10:40:36)',
'*KEYWORD',
'*ELEMENT_SOLID']
pattern2 = ['*NODE']
pattern3 = ['*END']

output_file_name = mesh_file_dir + "/test_volume.k"
if os.path.exists(output_file_name):
    os.remove(output_file_name)

if not os.path.exists(output_file_name):
    file = open(output_file_name, "w")
    file.close()
with open(output_file_name,'a+') as f:
    for i in range(len(pattern1)):
        f.write(pattern1[i])
        f.write('\n')
    for i in range(len(s_i)):
        f.write(s_i[i])
    for i in range(len(pattern2)):
        f.write(pattern2[i])
        f.write('\n')
    for i in range(len(v_i)):
        f.write(v_i[i])        
    for i in range(len(pattern3)):
        f.write(pattern3[i])
        f.write('\n')

