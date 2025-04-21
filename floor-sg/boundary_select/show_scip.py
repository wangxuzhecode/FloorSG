from matplotlib import pyplot as plt
import numpy as np
import glob
import os

# 打印所有的 .txt 文件名
def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）

fig, ax = plt.subplots()
color_lst = []

for i in range(1, 46):
    array = []
    txt_file = "C:\\2024\\FloorSG\\S3DIS_vector\\area1\\" + str(i) +"\\line.txt"
    print(txt_file)
    with open(txt_file, 'r') as file:
        for line in file:
            # 去掉行尾的换行符，并将每行的数字按空格分割成列表       C:\2024\FloorSG\S3DIS_res\10.txt
            row = list(map(float, line.strip().split()))
            array.append(row)
    cur_color = random_color()
    color_lst.append(cur_color)
    print(cur_color)
    for data in array:
        # print(cur_color)
        plt.plot([data[0], data[2]], [data[1], data[3]], color=cur_color, label='Split Segment')

# l = [j for j in range(0, 40)]
    # for i in l:
    #     if i in res_lst:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    #     else:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')


ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()




path_dir = r"C:\2024\FloorSG\S3DIS_area1\*.txt"

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
    cur_color = color_lst[i]
    for data in array:
        ax.plot([data[0], data[2]], [data[1], data[3]], color=cur_color, label='Split Segment', lw = 1)

# l = [j for j in range(0, 40)]
    # for i in l:
    #     if i in res_lst:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    #     else:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')

ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()