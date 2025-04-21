import math
import os
import numpy as np
import sys



def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def get_boundary_map(resized_image2):
    boundary_map = np.zeros_like(resized_image2)
    for i in range(resized_image2.shape[0]):
        for j in range(resized_image2.shape[1]):
            if resized_image2[i][j] != 0:
                for a, b in zip([1, 0, -1, 0, 1, -1, 1, -1], [0, 1, 0, -1, 1, -1, -1, 1]):
                    if (i + a < resized_image2.shape[0] and i + a >= 0) and (
                            j + b < resized_image2.shape[1] and j + b >= 0):
                        if resized_image2[i][j] != resized_image2[i + a][j + b]:
                            boundary_map[i][j] = 1
                            break
                    else:
                        boundary_map[i][j] = 1
    return boundary_map

area_name = r'E:\FloorSG\Expriment_output\Matterport3d\zsNo4HB9uLZ_filter_resize.txt'
img_orig = read_txt_to_matrix(area_name)
image = np.array(img_orig ,dtype=np.uint32)



for i in range(1, 28):
    print('第', i, '个房间')
    tmp_idx = np.argwhere(image == i)
    if tmp_idx.shape[0] == 0:
        continue
    new_image = np.zeros(shape=image.shape, dtype=np.uint32)
    new_image[image == i] = 255
    cur_boundary_map = get_boundary_map(new_image)

    image[cur_boundary_map==1]=0
np.savetxt(r'E:\FloorSG\Expriment_output\Matterport3d\zsNo4HB9uLZ_filter_resize.txt', image, fmt='%d')
