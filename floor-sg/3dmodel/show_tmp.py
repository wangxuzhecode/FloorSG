from matplotlib import pyplot as plt
import numpy as np
import glob
import os

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

path_dir = "C:\\2024\\FloorSG\\S3DIS_res\\*.txt"

txt_files = glob.glob(path_dir)

# 打印所有的 .txt 文件名


fig, ax = plt.subplots()
array = []
txt_file = "area4_wall.txt"
data = get_data(txt_file)

for i, room in enumerate(data):
    # if i!=9:
    #     continue
    x = room[::3].reshape(-1, 1)  # 提取 x 坐标
    y = room[1::3].reshape(-1, 1)  # 提取 y 坐标
    z = room[2::3].reshape(-1, 1)
    poly = np.hstack((x, y, z))  # 连接成 (x, y, z) 坐标
    hash_wall = [0 for _ in range(poly.shape[0])]
    # print(poly.shape[0])
    tmp = -1
    for j in range(poly.shape[0]):
        st_p = (poly[tmp, 0], poly[tmp, 1])
        ed_p = (poly[j, 0], poly[j, 1])
        tmp+=1
        ax.plot([st_p[0], ed_p[0]], [st_p[1], ed_p[1]], color='r', label='Split Segment')


# l = [j for j in range(0, 40)]
    # for i in l:
    #     if i in res_lst:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    #     else:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')


ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()