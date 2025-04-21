from matplotlib import pyplot as plt
import numpy as np
import glob
import os

path_dir = r"C:\2024\FloorSG\S3DIS_area2\*.txt"

txt_files = glob.glob(path_dir)

# 打印所有的 .txt 文件名


fig, ax = plt.subplots()

for i, txt_file in enumerate(txt_files):
    array = []
    # if i==0:
    #     continue
    with open(txt_file, 'r') as file:
        for line in file:
            # 去掉行尾的换行符，并将每行的数字按空格分割成列表
            row = list(map(float, line.strip().split()))
            array.append(row)
    for data in array:
        ax.plot([data[0], data[2]], [data[1], data[3]], color='b', label='Split Segment', lw = 1)


# txt_file = r"E:\FloorSG\code\S3DIS_total\matterport3d_model\scene2_door2.txt"
# with open(txt_file, 'r') as file:
#     for line in file:
#         # 去掉行尾的换行符，并将每行的数字按空格分割成列表       C:\2024\FloorSG\S3DIS_res\10.txt
#         row = list(map(float, line.strip().split()))
#         array.append(row)
# # for data in array:
#     ax.plot([data[0], data[2]], [data[1], data[3]], color='r', label='Split Segment')
# l = [j for j in range(0, 40)]
    # for i in l:
    #     if i in res_lst:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    #     else:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')

ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()