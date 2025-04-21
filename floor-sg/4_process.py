import numpy as np
import math
from scipy.ndimage import label
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
    img_work = np.array(img_work, np.uint32)

    for i in range(np.max(img_work)):
        idx = np.argwhere(img_work==i)
        # print(idx)
        if idx.shape[0]!=0:
            tmp = np.zeros_like(img_work)
            tmp[img_work==i]=1
            # print(tmp.shape)
            # print(label)
            labeled_matrix, num_features = label(tmp)
            unique_values = np.arange(1, num_features + 1)
            label_to_value = {label: value for label, value in zip(range(1, num_features + 1), unique_values)}
            segmented_matrix = np.zeros_like(tmp)
            for label1, value in label_to_value.items():
                segmented_matrix[labeled_matrix == label1] = value
            if np.max(segmented_matrix)==2:
                idx1 = np.argwhere(segmented_matrix==1)
                idx2 = np.argwhere(segmented_matrix==2)
                if idx1.shape[0]<idx2.shape[0]:
                    img_work[segmented_matrix==1]=0
                else:
                    img_work[segmented_matrix==2]=0
    output_path = sys.argv[2]
    save_matrix_to_txt(img_work, output_path)