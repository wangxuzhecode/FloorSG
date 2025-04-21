import numpy as np
import math
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes
import sys

from scipy.stats import bootstrap
from sympy import Matrix
from sympy.diffgeom import metric_to_Ricci_components


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

        upper_num = 0
        bottom_num = 0
        for ii in range(left, right+1):
            if matrix[upper, ii] != i:
                upper_num+=1
            if matrix[bottom, ii] != i:
                bottom_num+=1

        left_num=0
        right_num=0
        for ii in range(upper, bottom+1):
            if matrix[ii, left] != i:
                left_num+=1
            if matrix[ii, right]!=i:
                right_num+=1
        print(upper_num, bottom_num,left_num, right_num)
        if upper_num>theta:
            mask = cur_idx[cur_idx[:, 0] == upper]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0
        if bottom_num>theta:
            mask = cur_idx[cur_idx[:, 0] == bottom]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0
        if left_num>theta:
            mask = cur_idx[cur_idx[:, 1] == left]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0
        if right_num>theta:
            mask = cur_idx[cur_idx[:, 1] == right]
            for j in range(mask.shape[0]):
                matrix[mask[j][0]][mask[j][1]] = 0

    output_path = sys.argv[2]
    save_matrix_to_txt(matrix, output_path)
