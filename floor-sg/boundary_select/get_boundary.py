import math
import os
import sys

import numpy as np


def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def get_area_by_label(img_work, label):
    area_list = []
    for y in range(len(img_work)):
        for x in range(len(img_work[0])):
            cur_label = img_work[y][x]
            if cur_label == label:
                area_list.append((y, x))
    return area_list

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

def change_points(st, ed):
    if st[0]==ed[0]:
        if st[1]<ed[1]:
            res_st = st
            res_ed = ed
        else:
            res_st = ed
            res_ed = st
    else:
        if st[0]<ed[0]:
            res_st = st
            res_ed = ed
        else:
            res_st = ed
            res_ed = st
    return res_st, res_ed


def merge_lines(segments):
    """合并共线且重叠的线段"""
    merged_segments = []

    def is_overlap_or_connected(seg1, seg2):
        """检查两个线段是否重叠或连接"""

        def project_to_1d(p, seg):
            return min(seg[0], seg[1]) <= p <= max(seg[0], seg[1])

        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        if are_collinear(np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])) and \
                are_collinear(np.array([x1, y1]), np.array([x2, y2]), np.array([x4, y4])):
            return (project_to_1d(x3, (x1, x2)) and project_to_1d(y3, (y1, y2))) or \
                   (project_to_1d(x4, (x1, x2)) and project_to_1d(y4, (y1, y2)))
        return False

    def merge_two_segments(seg1, seg2):
        """合并两个重叠或连接的线段"""
        x1, y1, x2, y2 = seg1
        x3, y3, x4, y4 = seg2
        x_min = min(x1, x2, x3, x4)
        y_min = min(y1, y2, y3, y4)
        x_max = max(x1, x2, x3, x4)
        y_max = max(y1, y2, y3, y4)
        return (x_min, y_min, x_max, y_max)

    while segments:
        seg = segments.pop(0)
        merged = False
        for i in range(len(merged_segments)):
            if is_overlap_or_connected(seg, merged_segments[i]):
                merged_segments[i] = merge_two_segments(seg, merged_segments[i])
                merged = True
                break
        if not merged:
            merged_segments.append(seg)

    return merged_segments

img_name = r'C:\2024\FloorSG\ISPRS_result\MRF_filter_refine_filter_res_pad.txt'
img_work = read_txt_to_matrix(img_name)
floorplan = np.array(img_work, np.uint32)

point_path = 'C:\\2024\\data\\ISPRS\\TUB1\\all.txt'
matrix = []
with open(point_path, 'r') as file:
    for line in file:
        # 将每行的数字按空格或制表符分割，并转换为float类型，存储为一行矩阵
        row = list(map(float, line.split()))
        matrix.append(row)
pointcloud = np.array(matrix)[:, 0:3]

labelID = 1
gridsize = 0.12
min_data = [-9.393884660000000, -24.515441890000000]
height = 126
width = 356
label_map = np.zeros((height, width), dtype = np.int32)
boundary_map = get_boundary_map(floorplan)

for i in range(1, np.max(floorplan) + 1):
    print('第', i, '个房间')
    # if i!=17:
    #     continue
    # if i!=1 and i!=17 and i!=18 and i!=19 and i!=28 and i!=31 and i!=34 and i!=37 and i!=41 and i!=42 and i!=43 and i!=44:
    #     continue
    new_image = np.zeros(shape=floorplan.shape, dtype=np.uint32)
    new_image[floorplan == i] = 255
    cur_boundary_map = get_boundary_map(new_image)
    boundary_lst = []
    for x in range(cur_boundary_map.shape[0]):
        for y in range(cur_boundary_map.shape[1]):
            if cur_boundary_map[x][y] == 1:
                boundary_lst.append((x, y))
    num_edge = 0
    cur_pos = boundary_lst[0]
    hash_map = np.zeros_like(cur_boundary_map)
    segments = []
    while 1:
        if num_edge == len(boundary_lst):
            break
        # print(cur_pos)
        for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
            if (cur_pos[0] + a < floorplan.shape[0] and cur_pos[0] + a >= 0) and (
                    cur_pos[1] + b < floorplan.shape[1] and cur_pos[1] + b >= 0):
                if cur_boundary_map[cur_pos[0] + a][cur_pos[1] + b] == 1 and hash_map[cur_pos[0] + a][
                    cur_pos[1] + b] == 0:
                    res_st, res_ed = change_points(cur_pos, (cur_pos[0] + a, cur_pos[1] + b))
                    # with open(roomline_path, "a") as f:
                    #     f.write(f"{res_st[0]}\t{res_st[1]}\t{res_ed[0]}\t{res_ed[1]}\t{11}\t{11}\n")
                    segments.append((res_st[0], res_st[1], res_ed[0], res_ed[1]))
                    hash_map[cur_pos[0] + a][cur_pos[1] + b] = 1
                    cur_pos = (cur_pos[0] + a, cur_pos[1] + b)
                    num_edge += 1
                    break

    merged_segments = merge_lines(segments)
print(merged_segments.shape)
# selected_points = []
#
# for i in range(pointcloud.shape[0]):
#     x = pointcloud[i, 0]
#     y = pointcloud[i, 1]
#     cur_height = math.ceil((x - min_data[0]) / gridsize)
#     cur_width = math.ceil((y - min_data[1]) / gridsize)
#     cur_height = max(cur_height, 1)
#     cur_width = max(cur_width, 1)
#
#     if boundary_map[cur_height - 1, cur_width - 1] == 1:
#         selected_points.append(pointcloud[i])
#
# selected_points = np.array(selected_points)
# print(selected_points.shape)
# np.savetxt('boundary.txt', selected_points)
