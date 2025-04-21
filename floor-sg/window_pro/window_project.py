import copy
import glob
import pyvista as pv
import numpy as np
from matplotlib import pyplot as plt

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
        # print(slope_diff, slope_diff2)
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
    # print(closest_segment,closest_distance)
    # 如果找到了最近的线段，将待投影线段投影到该线段上
    if closest_segment is not None and closest_distance<1:
        # print(closest_distance)
        proj_start = project_point_on_line(line[0], closest_segment[0], closest_segment[1])
        proj_end = project_point_on_line(line[1], closest_segment[0], closest_segment[1])

        proj_start_clamped = clamp_projection(proj_start, closest_segment[0], closest_segment[1])
        proj_end_clamped = clamp_projection(proj_end, closest_segment[0], closest_segment[1])

        return (proj_start_clamped, proj_end_clamped), selected_idx

    return None, None

def get_data(in_path):
    with open(in_path, 'r') as f:
        data = f.readlines()
        data = [(x.strip('\n').split(' ')) for x in data]
        new_data = []
        for i in data:
            ps = []
            for j in i:
                if j != '':
                    ps.append(float(j))
            new_data.append(np.array(ps))
    return new_data

fig, ax = plt.subplots()
doorpath = 'area4_window.txt'
doors = get_data(doorpath)

path_dir = r"E:\FloorSG\Expriment_output\S3DIS\wall_vector\*.txt"
txt_files = glob.glob(path_dir)
wall_line = []
for i, txt_file in enumerate(txt_files):
    cur_array = []
    with open(txt_file, 'r') as file:
        for line in file:
            # 去掉行尾的换行符，并将每行的数字按空格分割成列表
            row = list(map(float, line.strip().split()))
            cur_array.append(row)
    for data in cur_array:
        wall_line.append(((data[0], data[1]), (data[2], data[3])))
        ax.plot([data[0], data[2]], [data[1], data[3]], color='b', label='Split Segment', lw = 1)


height = 2
with open('area4_project_window.txt', 'w', encoding='utf-8') as f:
    for i, line in enumerate(doors):
        # if i!=20 and i!=23:
        #     continue
        cur_line = ((line[0], line[1]),(line[2], line[3]))
        # ax.plot([line[0], line[2]], [line[1], line[3]], color='k', label='Split Segment', lw = 1)
        projected_line, selected_idx = project_line_to_closest_segment(wall_line, cur_line)
        # print(projected_line)
        if selected_idx==None:
            continue
        sel_line = copy.deepcopy(wall_line[selected_idx])
        del wall_line[selected_idx]
        if selected_idx==None:
            continue

        plt.plot([projected_line[0][0], projected_line[1][0]], [projected_line[0][1], projected_line[1][1]], 'r', label='Extended Line')
        cur_str = str(projected_line[0][0]) + " " + str(projected_line[0][1]) + " " + str(height) + " " + str(projected_line[1][0]) + " " + str(projected_line[1][1]) + " " + str(height)
        f.writelines(cur_str)
        f.write('\n')
        projected_line2, selected_idx2 = project_line_to_closest_segment(wall_line, cur_line)
        wall_line.append(sel_line)
        if selected_idx2 is None:
            continue
        plt.plot([projected_line2[0][0], projected_line2[1][0]], [projected_line2[0][1], projected_line2[1][1]], 'r', label='Extended Line')


        cur_str = str(projected_line2[0][0]) + " " + str(projected_line2[0][1]) + " " + str(height) + " " + str(projected_line2[1][0]) + " " + str(projected_line2[1][1]) + " " + str(height)
        f.writelines(cur_str)
        f.write('\n')

ax.set_aspect('equal')  # 设置坐标轴比例相等
ax.set_axis_off()  # 关闭坐标轴显示
plt.show()