import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import math

from collections import defaultdict


# 计算连通分量的DFS
def dfs(graph, node, visited, component):
    visited.add(node)
    component.append(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, component)


# 主函数：获取闭合图形
def find_closed_shapes(segments):
    # 构建图（字典形式）
    graph = defaultdict(list)
    for p1, p2 in segments:
        graph[p1].append(p2)
        graph[p2].append(p1)

    visited = set()
    closed_shapes = []

    # 查找所有连通分量
    for node in graph:
        if node not in visited:
            component = []
            dfs(graph, node, visited, component)

            # 检查是否形成闭合图形（可以简单地检查所有端点是否相连形成环）
            if len(component) >= 3:  # 至少是一个三角形
                closed_shapes.append(component)
    return closed_shapes

# 定义浮点数比较的容差值
EPSILON = 1e-2


def are_close(a, b, epsilon=EPSILON):
    """判断两个浮点数是否相等，考虑误差范围"""
    return abs(a - b) < epsilon


def can_merge(segment1, segment2):
    """判断两条线段是否可以合并"""
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    # 判断斜率是否相等，避免浮点数误差
    # (y2 - y1) / (x2 - x1) == (y4 - y3) / (x4 - x3)
    # 改写为交叉相乘，避免除法
    if not are_close((y2 - y1) * (x4 - x3), (y4 - y3) * (x2 - x1)):
        return False

    # 判断是否首尾相连
    if are_close(x2, x3) and are_close(y2, y3):
        return True
    if are_close(x1, x4) and are_close(y1, y4):
        return True
    return False


def merge_segments(segment1, segment2):
    """合并两条线段"""
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    # 合并线段，取最小的起点和最大的终点
    if are_close(x2, x3) and are_close(y2, y3):
        return ((x1, y1), (x4, y4))
    elif are_close(x1, x4) and are_close(y1, y4):
        return ((x3, y3), (x2, y2))
    else:
        return None  # 如果不能合并，返回 None


def merge_all_segments(segments):
    """合并所有可以合并的线段"""
    merged_segments = []

    while segments:
        segment = segments.pop(0)  # 取出一个线段进行合并
        merged = False

        for i, other in enumerate(merged_segments):
            if can_merge(segment, other):
                # 如果可以合并，合并并更新
                merged_segments[i] = merge_segments(segment, other)
                merged = True
                break

        if not merged:
            # 如果没有合并成功，直接加入新的线段
            merged_segments.append(segment)

    return merged_segments

def check_walls(walls, p1, p2, i):
    f1 = False
    f2 = False
    for j, (p3, p4) in enumerate(walls):
        if i==j:continue
        if (p2==p3) or (p2==p4):
            f2 = True
        if (p1==p3) or (p1==p4):
            f1 = True
    if f1==False:
        for j, (p3, p4) in enumerate(walls):
            if i==j:
                continue
            if math.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)<0.01:
                print(p1, p3)
                walls[i] = (p3, p2)
                print(1111)
            elif math.sqrt((p1[0]-p4[0])**2+(p1[1]-p4[1])**2)<0.01:
                walls[i] = (p4, p2)
                print(1111)
    if f2 ==False:
        for j, (p3, p4) in enumerate(walls):
            if math.sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)<0.01:
                walls[i] = (p1, p3)
                print(p2, p3)
            elif math.sqrt((p2[0]-p4[0])**2+(p2[1]-p4[1])**2)<0.01:
                walls[i] = (p1, p4)
                print(p2, p4)

    return walls

def check(walls):
    num = len(walls)
    i=0
    while i<num:
        p1,p2=walls[i]
        walls = check_walls(walls, p1, p2, i)
        i+=1
    return walls

path_dir = "C:\\2024\\FloorSG\\S3DIS_area4\\"
# txt_files = glob.glob(path_dir)
output_path = 'E:\\FloorSG\\Expriment_output\\S3DIS\\wall_vector\\'
out_path = 'tmp.txt'
height = 2.2

with open(out_path, 'w', encoding='utf-8') as f1:
    for i in range(1, 49):
        # if i !=10:
        #     continue
        txt_file = os.path.join(path_dir, str(i)+'.txt')

        cur_array = []
        wall_line = []
        with open(txt_file, 'r') as file:
            for line in file:
                # 去掉行尾的换行符，并将每行的数字按空格分割成列表
                row = list(map(float, line.strip().split()))
                cur_array.append(row)
        for data in cur_array:
            wall_line.append(((data[0], data[1]), (data[2], data[3])))

        closed_shapes = find_closed_shapes(wall_line)
        print(len(closed_shapes))
        sel_idx = 0
        max_shape = -1
        for j, shape in enumerate(closed_shapes):
            if len(shape)>max_shape:
                max_shape = len(shape)
                sel_idx = j
        cur_walls = []
        tmp = -1
        for j, node in enumerate(closed_shapes[sel_idx]):
            cur_walls.append((closed_shapes[sel_idx][tmp], node))
            tmp+=1


        # print(len(cur_walls))

        num1= len(cur_walls)
        merged = merge_all_segments(cur_walls)
        num2= len(merged)
        merged2 = merge_all_segments(merged)
        num3= len(merged2)
        merged3 = merge_all_segments(merged2)
        merged4 = merge_all_segments(merged3)
        merged5 = merge_all_segments(merged4)
        merged5 = check(merged5)
        closed_shapes = find_closed_shapes(merged5)
        print(len(merged5), len(closed_shapes[0]))
        # print(len(closed_shapes))
        ## 应当保存合并之后的
        for j, node in enumerate(closed_shapes[0]):
            cur_str = str(node[0]) + " " + str(node[1]) + " " + str(height)
            if j != len(closed_shapes[sel_idx]) - 1:
                cur_str += " "
            f1.writelines(cur_str)
        f1.write('\n')

        with open(os.path.join(output_path, os.path.basename(txt_file)), 'w', encoding='utf-8') as f:
            for line in merged5:
                cur_str = str(line[0][0]) + " " + str(line[0][1]) + " " + str(line[1][0]) + " " + str(line[1][1])
                f.writelines(cur_str)
                f.write('\n')