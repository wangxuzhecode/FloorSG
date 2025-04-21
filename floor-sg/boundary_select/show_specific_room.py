from matplotlib import pyplot as plt
import numpy as np
import glob
import os

path_dir = "C:\\2024\\FloorSG\\S3DIS_res\\*.txt"

txt_files = glob.glob(path_dir)

# 打印所有的 .txt 文件名


fig = plt.figure(figsize=(10, 7))
axes2 = fig.add_subplot(1, 1, 1)
array = []
txt_file = r"E:\FloorSG\MRF_test\test_109\109_vector\1\line.txt"
with open(txt_file, 'r') as file:
    for line in file:
        # 去掉行尾的换行符，并将每行的数字按空格分割成列表       C:\2024\FloorSG\S3DIS_res\10.txt
        row = list(map(float, line.strip().split()))
        array.append(row)
for data in array:
    plt.plot([data[0], data[2]], [data[1], data[3]], color='r', label='Split Segment')

# l = [j for j in range(0, 40)]
    # for i in l:
    #     if i in res_lst:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    #     else:
    #         plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')


plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()