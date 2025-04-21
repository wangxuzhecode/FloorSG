import pickle
import scipy
from matplotlib import pyplot as plt
import numpy as np
import glob
import os


# 打印所有的 .txt 文件名
def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）

def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def get_color_lst(mat_path, floor_path):
    color_lst = []
    mat_data = scipy.io.loadmat(mat_path)
    color = mat_data['Lrgb']
    img_work = read_txt_to_matrix(floor_path)
    floorplan = np.array(img_work, np.uint32)
    for i in range(1, np.max(floorplan)+1):
        idx = np.argwhere(floorplan==i)
        if idx.shape[0]==0:
            continue
        cur_color = color[idx[0,0], idx[0,1]]
        color_lst.append(cur_color)
    return color_lst

def get_data(in_path):
    with open(in_path, 'r') as f:
        data = f.readlines()
        data = [(x.strip('\n').split(' ')) for x in data]
        new_data = []
        for i in data:
            ps = []
            for j in i:
                if j != '':
                    ps.append(float(j))
            new_data.append(np.array(ps))
    return new_data


color_lst = get_color_lst(r'C:\2024\Code\FloorSG\Seg\scene1_color.mat',
                          r'E:\FloorSG\Expriment_output\Matterport3d\JeFG25nYj2p_filter.txt')

fig, ax = plt.subplots()
array = []
txt_file = "scene1_wall.txt"
data = get_data(txt_file)

for i, room in enumerate(data):
    # print(i)
    # if i !=5:
    #     continue
    x = room[::3].reshape(-1, 1)  # 提取 x 坐标
    y = room[1::3].reshape(-1, 1)  # 提取 y 坐标
    z = room[2::3].reshape(-1, 1)
    poly = np.hstack((x, y, z))  # 连接成 (x, y, z) 坐标
    hash_wall = [0 for _ in range(poly.shape[0])]
    print(poly.shape[0], i)
    tmp = -1
    # if i>len(color_lst)-1:
    #     cur_color = color_lst[5]
    # else:
    #     cur_color = color_lst[i]
    cur_color = color_lst[i]

    for j in range(poly.shape[0]):
        st_p = (poly[tmp, 0], poly[tmp, 1])
        ed_p = (poly[j, 0], poly[j, 1])
        tmp+=1
        ax.plot([st_p[0], ed_p[0]], [st_p[1], ed_p[1]], color=cur_color/255, label='Split Segment', linewidth=1.5)


# l = [j for j in range(0, 40)]
    # for i in l:
    #     if i in res_lst:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    #     else:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')


ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()