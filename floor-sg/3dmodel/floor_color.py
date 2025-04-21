import numpy as np
import scipy.io


def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix
# 读取 .mat 文件
mat_data = scipy.io.loadmat(r'C:\2024\Code\FloorSG\Seg\area4_color.mat')

# 如果你知道文件中存储的变量名，可以直接访问
# variable_name = 'your_variable'  # 例如，文件中存储的变量名为 'your_variable'
# if variable_name in mat_data:
#     variable_data = mat_data[variable_name]
    # print(variable_data)
data = mat_data['Lrgb']

print(data.shape)

img_work = read_txt_to_matrix(r'E:\FloorSG\Expriment_output\S3DIS\Boundary_optim\area4.txt')
floorplan = np.array(img_work, np.uint32)
print(floorplan.shape)

for i in range(1, np.max(floorplan)+1):
    idx = np.argwhere(floorplan==i)
    print(idx.shape)
    cur_color = data[idx[0,0], idx[0,1]]
    print(cur_color.shape)