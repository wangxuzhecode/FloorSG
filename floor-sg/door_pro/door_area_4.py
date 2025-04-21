import glob

import numpy as np
import os
import math
import matplotlib.pyplot as plt
import math


# 计算两点之间的欧几里得距离
def distance_point_to_point(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 计算斜率与x轴的夹角（单位：度）
def calculate_angle_with_x_axis(x1, y1, x2, y2):
    if x2 == x1:  # 垂直线段，斜率为无穷大，夹角为90度
        return 90
    # 计算斜率
    slope = (y2 - y1) / (x2 - x1)
    # 计算与x轴的夹角（单位：弧度）
    angle_rad = math.atan(slope)
    # 转换为角度
    angle_deg = math.degrees(angle_rad)
    return angle_deg


# 计算两条线段之间的最短距离
def distance_between_segments(x1, y1, x2, y2, x3, y3, x4, y4):
    def point_to_segment_distance(px, py, ax, ay, bx, by):
        # 向量AP = (px - ax, py - ay), 向量AB = (bx - ax, by - ay)
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_dot_ap = abx * apx + aby * apy  # 向量点积
        ab_dot_ab = abx * abx + aby * aby  # 向量的自点积
        t = ab_dot_ap / ab_dot_ab

        if t < 0:
            closest_x, closest_y = ax, ay  # 投影点在A左侧
        elif t > 1:
            closest_x, closest_y = bx, by  # 投影点在B右侧
        else:
            closest_x = ax + t * abx
            closest_y = ay + t * aby

        return distance_point_to_point(px, py, closest_x, closest_y)

    # 计算四种端点到线段的距离
    return min(
        point_to_segment_distance(x1, y1, x3, y3, x4, y4),
        point_to_segment_distance(x2, y2, x3, y3, x4, y4),
        point_to_segment_distance(x3, y3, x1, y1, x2, y2),
        point_to_segment_distance(x4, y4, x1, y1, x2, y2)
    )


# 判断两条线段是否满足斜率与距离范围
def are_segments_similar(x1, y1, x2, y2, x3, y3, x4, y4, angle_threshold=10.0, distance_threshold=5.0):
    # 计算两条线段与x轴的夹角
    angle1 = calculate_angle_with_x_axis(x1, y1, x2, y2)
    angle2 = calculate_angle_with_x_axis(x3, y3, x4, y4)

    # 斜率与x轴夹角的差异
    if abs(angle1 - angle2) > angle_threshold:
        return False

    # 计算两条线段之间的最短距离
    min_distance = distance_between_segments(x1, y1, x2, y2, x3, y3, x4, y4)

    # 如果最短距离大于阈值，则认为两条线段不接近
    if min_distance > distance_threshold:
        return False

    return True


def distance(point, m, b):
    """计算点到直线的距离"""
    x, y = point
    return abs(-m * x + 1 * y - b) / np.sqrt(m ** 2 + 1)

def ransac_line_fitting(points, iterations=2000, threshold=0.05):
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

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        # if abs(min(m, 1/(m+1e-9))) > 1 / 20:  # 约束斜率
        #     continue
        if m==0:
            continue
        inliers = [point for point in points if distance(point, m, b) < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m = m
            best_b = b

    return best_m, best_b, np.array(best_inliers)

def fit_line(points, thres, original_point,out=0.2):
    m, b, _ = ransac_line_fitting(points, threshold=thres)
    x_fit = np.linspace(np.min(original_point, 0)[0], np.max(points, 0)[0], 100)
    y_fit = m * x_fit + b
    y_min = np.min(original_point, 0)[1]
    y_max = np.max(original_point, 0)[1]
    if max(y_fit[-1], y_fit[0])>y_max+0.6 or min(y_fit[-1], y_fit[0])<y_min-0.6:
        new_x_min = (y_min - b) / m
        new_x_max = (y_max - b) / m
        st_pos = (new_x_min, y_min)
        ed_pos = (new_x_max, y_max)
    else:
        st_pos = (np.min(original_point, 0)[0], y_fit[0])
        ed_pos = (np.max(original_point, 0)[0], y_fit[-1])
    outliers = np.array([point for point in points if distance(point, m, b) > out])
    return st_pos, ed_pos, outliers

def get_bbox(points):
    x_min = np.min(points, 0)[0]
    x_max = np.max(points, 0)[0]
    y_min = np.min(points, 0)[1]
    y_max = np.max(points, 0)[1]
    x_length = x_max - x_min
    y_length = y_max - y_min
    return x_length, y_length

def fit_one_door(points):
    line_lst = []
    x_length, y_length = get_bbox(points)
    st_pos1, ed_pos1, outliers1 = fit_line(points, thres=0.1,original_point=points, out=0.25)
    # ax.scatter(points[:,0], points[:,1], color='k',s=1)

    # ax.scatter(outliers1[:,0], outliers1[:,1], color='r',s=1)

    line_lst.append((st_pos1, ed_pos1))
    # print(y_length,x_length)
    if min(y_length, x_length)>0.6:
        st_pos2, ed_pos2, outliers2 = fit_line(outliers1, thres=0.02,original_point=points)
        # ax.scatter(outliers2[:, 0], outliers2[:, 1], color='y', s=1)

        line_lst.append((st_pos2, ed_pos2))
        if outliers2.shape[0]>15:
            st_pos3, ed_pos3, outliers3 = fit_line(outliers2, thres=0.03,original_point=points)
            if are_segments_similar(st_pos1[0], st_pos1[1], ed_pos1[0], ed_pos1[1], st_pos3[0], st_pos3[1], ed_pos3[0], ed_pos3[1], 2, 1)==False:
                if are_segments_similar(st_pos2[0], st_pos2[1], ed_pos2[0], ed_pos2[1], st_pos3[0], st_pos3[1], ed_pos3[0], ed_pos3[1], 2, 1)==False:
                    line_lst.append((st_pos3, ed_pos3))

    return line_lst

doorpath = r'E:\Stanford3dDataset_v1.2\area4_single_door_sub'
# fig = plt.figure(figsize=(10, 7))
fig, ax = plt.subplots()

files = os.listdir(doorpath)
print(files)

door_line = []
for i, filename in enumerate(files):
    if i!=18:
        continue
    print(i, filename)
    matrix = []
    with open(os.path.join(doorpath, filename), 'r') as file:
        for line in file:
            # 将每行的数字按空格或制表符分割，并转换为float类型，存储为一行矩阵
            row = list(map(float, line.split()))
            matrix.append(row)
    pointcloud_per_room = np.array(matrix)[:, 0:3]
    print(pointcloud_per_room.shape)
    idx = np.argwhere((pointcloud_per_room[:,2]>1.8))
    points = pointcloud_per_room[idx]
    print(points.shape)
    points = points[:, 0, 0:2]
    line_lst = fit_one_door(points)
    for line in line_lst:
        ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='r', label='Split Segment', lw = 2)

    ax.scatter(points[:,0], points[:,1], color='k',s=1)
    # if i>10:break

ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()