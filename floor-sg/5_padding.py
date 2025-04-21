import numpy as np
import math
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes
import sys

def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix
def save_matrix_to_txt(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            row_str = ' '.join(map(str, row))  # 将每行转换为字符串，并用制表符分隔
            file.write(row_str + '\n')  # 写入每行数据到文件

if __name__ == '__main__':

    img_name = sys.argv[1]
    img_work = read_txt_to_matrix(img_name)
    matrix = np.array(img_work, np.uint32)

    # 获取唯一的区域标识符（假设0是空白区域）
    regions = np.unique(matrix)
    regions = regions[regions != 0]  # 去除空白区域

    # 初始化结果矩阵
    filled_matrix = np.copy(matrix)

    # 对每个区域进行填充
    for region in regions:
        # 创建二值图像，区域值为1，其他为0
        # print(region)
        # if region==6:
        #     continue
        binary_image = (matrix == region).astype(int)
        #
        # 填充孔洞
        filled_binary_image = binary_fill_holes(binary_image)

        # 将填充后的图像赋回到结果矩阵中
        filled_matrix[filled_binary_image] = region

    matrix = filled_matrix

    num = 1
    num_segmented = np.max(matrix)

    for object_id in range(1, num_segmented + 1):
        rows, cols = np.where(matrix == object_id)
        print(f"Object ID: {object_id}, Number of Occurrences: {len(rows)}")

        if len(rows) > 0:
            for i in range(len(rows)):
                matrix[rows[i], cols[i]] = num
            num += 1

    theta = int(sys.argv[3])

    for i in range(1, np.max(matrix)+1):

        cur_idx = np.argwhere(matrix==i)
        upper = np.min(cur_idx,0)[0]
        bottom = np.max(cur_idx,0)[0]
        left = np.min(cur_idx,0)[1]
        right = np.max(cur_idx,0)[1]
        count_upper = np.sum(cur_idx[:, 0] == upper)
        count_bottom = np.sum(cur_idx[:, 0] == bottom)
        count_left = np.sum(cur_idx[:, 1] == left)
        count_right = np.sum(cur_idx[:, 1] == right)
        if count_upper <=theta:
            mask = cur_idx[cur_idx[:, 0] == upper]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0

        if count_bottom <=theta:
            mask = cur_idx[cur_idx[:, 0] == bottom]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0

        if count_left <=theta:
            mask = cur_idx[cur_idx[:, 1] == left]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0

        if count_right <=theta:
            mask = cur_idx[cur_idx[:, 1] == right]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0

    output_path = sys.argv[2]
    save_matrix_to_txt(matrix, output_path)
