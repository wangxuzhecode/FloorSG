import glob

import numpy as np
import os
import math
import matplotlib.pyplot as plt


# 计算两点之间的距离
def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# 计算线段的斜率
def calculate_slope(p1, p2):
    if p2[0] - p1[0] == 0:
        return np.inf  # 竖直线段
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


# 计算向量点积
def dot_product(v1, v2):
    return np.dot(v1, v2)


# 投影点计算
def project_point_on_line(point, line_start, line_end):
    line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
    point_vec = np.array([point[0] - line_start[0], point[1] - line_start[1]])

    # 投影公式
    proj_length = dot_product(point_vec, line_vec) / dot_product(line_vec, line_vec)
    proj_point = np.array([line_start[0], line_start[1]]) + proj_length * line_vec
    return proj_point


# 确保投影点在给定线段的范围内
def clamp_projection(proj_point, line_start, line_end):
    proj_x = np.clip(proj_point[0], min(line_start[0], line_end[0]), max(line_start[0], line_end[0]))
    proj_y = np.clip(proj_point[1], min(line_start[1], line_end[1]), max(line_start[1], line_end[1]))
    return np.array([proj_x, proj_y])


# 计算待投影线段的斜率
def get_line_slope(line):
    return calculate_slope(line[0], line[1])


# 计算待投影线段的端点
def project_line_to_closest_segment(segments, line):
    # 计算待投影线段的斜率
    line_slope = get_line_slope(line)

    # 找到斜率最接近且距离最近的线段
    closest_segment = None
    closest_distance = float('inf')
    selected_idx = -1
    for i, segment in enumerate(segments):
        segment_slope = get_line_slope(segment)

        # 计算斜率差异，选择斜率最接近的线段
        slope_diff = abs(line_slope - segment_slope)
        slope_diff2 = abs(1/(line_slope+1e-8) - 1/ (segment_slope+1e-8))
        # 计算投影结果和当前线段之间的距离
        proj_start = project_point_on_line(line[0], segment[0], segment[1])
        proj_end = project_point_on_line(line[1], segment[0], segment[1])

        proj_start_clamped = clamp_projection(proj_start, segment[0], segment[1])
        proj_end_clamped = clamp_projection(proj_end, segment[0], segment[1])

        dist_start = distance(proj_start_clamped, line[0])
        dist_end = distance(proj_end_clamped, line[1])
        avg_distance = (dist_start + dist_end) / 2  # 计算平均距离

        # 选择斜率最接近且距离最短的线段
        if min(slope_diff, slope_diff2) < 1 and avg_distance < closest_distance:  # 设定一个阈值来筛选斜率差异
            closest_distance = avg_distance
            closest_segment = segment
            selected_idx = i
    # 如果找到了最近的线段，将待投影线段投影到该线段上
    if closest_segment is not None and closest_distance<0.5:
        print(closest_distance)
        proj_start = project_point_on_line(line[0], closest_segment[0], closest_segment[1])
        proj_end = project_point_on_line(line[1], closest_segment[0], closest_segment[1])

        proj_start_clamped = clamp_projection(proj_start, closest_segment[0], closest_segment[1])
        proj_end_clamped = clamp_projection(proj_end, closest_segment[0], closest_segment[1])

        return (proj_start_clamped, proj_end_clamped), selected_idx

    return None, None

def m_distance(point, m, b):
    """计算点到直线的距离"""
    x, y = point
    return abs(-m * x + 1 * y - b) / np.sqrt(m ** 2 + 1)

def ransac_line_fitting(points, iterations=3000, threshold=0.03):
    best_inliers = []
    best_m = None
    best_b = None

    for _ in range(iterations):
        # 随机选择两个点
        sample = points[np.random.choice(points.shape[0], 2, replace=False)]
        (x1, y1), (x2, y2) = sample

        # 计算斜率和截距
        if x1 == x2:  # 处理垂直线的情况
            continue
        if y1 == y2:
            continue

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        # if abs(min(m, 1/(m+1e-9))) > 1 / 20:  # 约束斜率
        #     continue
        if m==0:
            continue
        inliers = [point for point in points if m_distance(point, m, b) < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m = m
            best_b = b

    return best_m, best_b, np.array(best_inliers)


class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.p1 = np.array([x1, y1])  # 第一个端点
        self.p2 = np.array([x2, y2])  # 第二个端点

    # 判断点P是否在两端点A和B之间
    def is_point_on_segment(self, P):
        # 判断P是否在线段AB上，要求P在A到B的范围内
        min_x, max_x = min(self.p1[0], self.p2[0]), max(self.p1[0], self.p2[0])
        min_y, max_y = min(self.p1[1], self.p2[1]), max(self.p1[1], self.p2[1])
        return min_x <= P[0] <= max_x and min_y <= P[1] <= max_y

    # 计算向量叉积，判断两线段是否相交
    def cross_product(self, A, B, C):
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    # 判断两线段是否相交
    def is_intersect(self, other):
        # 判断长线段self与短线段other是否相交
        # 使用叉积算法判断两线段是否相交
        # 假设 self.p1, self.p2 是长线段，other.p1, other.p2 是短线段

        p1, p2 = self.p1, self.p2
        p3, p4 = other.p1, other.p2

        # 计算长线段的叉积
        d1 = self.cross_product(p3, p4, p1)
        d2 = self.cross_product(p3, p4, p2)
        d3 = self.cross_product(p1, p2, p3)
        d4 = self.cross_product(p1, p2, p4)

        # 检查两线段是否有交点
        if d1 * d2 < 0 and d3 * d4 < 0:
            return True
        # 特殊情况：共线且部分重叠
        if d1 == 0 and self.is_point_on_segment(p3):
            return True
        if d2 == 0 and self.is_point_on_segment(p4):
            return True
        if d3 == 0 and other.is_point_on_segment(p1):
            return True
        if d4 == 0 and other.is_point_on_segment(p2):
            return True

        return False


# 将长线段根据短线段的端点进行分割
def split_line_by_short_segment(long_line, short_line):
    points = []

    # 检查短线段的两个端点是否在长线段上
    if long_line.is_intersect(short_line):
        points.append(short_line.p1)
        points.append(short_line.p2)
    if len(points) > 0:
        # 按照 x 和 y 坐标进行排序
        points_sorted = sorted(points, key=lambda p: (p[0], p[1]))

    # 分割长线段
    split_segments = []
    prev_point = long_line.p1
    for p in points_sorted:
        split_segments.append(LineSegment(prev_point[0], prev_point[1], p[0], p[1]))
        prev_point = p

    split_segments.append(LineSegment(prev_point[0], prev_point[1], long_line.p2[0], long_line.p2[1]))

    return split_segments

fig, ax = plt.subplots()
doorpath = r'E:\Stanford3dDataset_v1.2\area5_window'
# fig = plt.figure(figsize=(10, 7))

output_path = 'area5_window.txt'
files = os.listdir(doorpath)
print(files)

door_line = []
window_line = []
for i, filename in enumerate(files):
    print(i, filename)
    # if i!=46:continue
    matrix = []
    with open(os.path.join(doorpath, filename), 'r') as file:
        for line in file:
            # 将每行的数字按空格或制表符分割，并转换为float类型，存储为一行矩阵
            row = list(map(float, line.split()))
            matrix.append(row)
    # print(os.path.join(doorpath, filename))
    pointcloud_per_room = np.array(matrix)[:, 0:2]
    # print(pointcloud_per_room.shape)

    # if i <10:
    #     pointcloud_per_room[:, 0] -= 0.2
    # if i==27 or i==15 or i==31 or i==25 or i==26 or i==13 or i==20 or i==24 or i==16 or i==17 or i==18 or i==30 or i==6:
    #     pointcloud_per_room[:, 0] += 0.2

    m,b, _ = ransac_line_fitting(pointcloud_per_room)
    plt.scatter(pointcloud_per_room[:, 0], pointcloud_per_room[:, 1],s=1 ,color='black')  # 交点

    x_min = np.min(pointcloud_per_room, 0)[0]
    x_max = np.max(pointcloud_per_room, 0)[0]
    y_min = np.min(pointcloud_per_room, 0)[1]
    y_max = np.max(pointcloud_per_room, 0)[1]

    if x_max-x_min < y_max-y_min:
        y_fit = np.linspace(y_min, y_max, 100)
        x_fit = (y_fit - b) / (m)
        start_point = np.array([x_fit[0], y_min])
        end_point = np.array([x_fit[-1], y_max])
        # print(start_point, end_point)
        door_line.append((start_point, end_point))
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'r--', label='Extended Line')
    else:
        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = m * x_fit + b

        start_point = np.array([x_min, y_fit[0]])
        end_point = np.array([x_max, y_fit[-1]])
        # print(start_point, end_point)
        door_line.append((start_point, end_point))
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'r', label='Extended Line')


with open(output_path, 'w', encoding='utf-8') as f:
    for line in door_line:
        cur_str = str(line[0][0]) + " " + str(line[0][1]) + " " + str(line[1][0]) + " " + str(line[1][1])
        f.writelines(cur_str)
        f.write('\n')

ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()