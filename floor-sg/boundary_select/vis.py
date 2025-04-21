from matplotlib import pyplot as plt
import numpy as np


fig = plt.figure(figsize=(10, 7))
axes2 = fig.add_subplot(1, 1, 1)

# 打开文件并按行读取
array = []
with open(r'C:\2024\FloorSG\S3DIS_vector\area1\1\line.txt', 'r') as file:
    for line in file:
        # 去掉行尾的换行符，并将每行的数字按空格分割成列表
        row = list(map(float, line.strip().split()))
        array.append(row)

# 输出结果
print(array)
i = 2
j = 1


# plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
# plt.plot([array[j][0], array[j][2]], [array[j][1], array[j][3]], color='b', label='Split Segment')

res_lst =[


0,
1,
2,
18,
19,
20,



]

# res_lst = [1, 4,8,12,15,19 ,22, 25, 28, 31, 34, 38]
# res_lst = [11,12,13 ,20,21,22, 23]
l = [j for j in range(0, 24)]
for i in l:
    if i in res_lst:
        plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='r', label='Split Segment')
    # else:
        # plt.plot([array[i][0], array[i][2]], [array[i][1], array[i][3]], color='b', label='Split Segment')



plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()