import sys
import numpy as np


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
    input_path = sys.argv[1]
    image = read_txt_to_matrix(input_path)
    image = np.array(image, np.uint32)
    output_path = sys.argv[1]
    merge_label = int(sys.argv[2])
    image[image>100] = merge_label
    save_matrix_to_txt(image, output_path)