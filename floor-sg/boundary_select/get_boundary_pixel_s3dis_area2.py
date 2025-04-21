import math
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
from shapely.geometry import LineString
from decimal import Decimal, getcontext

# 设置高精度
getcontext().prec = 50  # 设置精度为50位

def read_txt_to_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # 假设每行数据以空格分隔
            matrix.append((row))
    return matrix

def convert_format(points):
    converted_points = []
    for point in points:
        x, y = point[0]  # point 是形如 [[x y]] 的列表
        converted_points.append((x, y))
    return converted_points

def find_nearest_point(x, y, points):
    min_distance = float('inf')  # 初始设为无穷大
    nearest_point = None
    for point in points:
        px, py = point
        distance = math.sqrt((px - x) ** 2 + (py - y) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
    return nearest_point

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

def are_collinear(p1, p2, p3):
    """检查点 p1, p2, p3 是否共线"""
    return np.isclose(np.cross(p2 - p1, p3 - p1), 0)


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

def to_decimal(point):
    """将普通浮点数转换为高精度Decimal"""
    return (Decimal(point[0]), Decimal(point[1]))

def find_intersection_between_lines(segments):

    intersection_points = set()
    shapely_segments = []
    for (x1, y1, x2, y2) in segments:
        # 使用Decimal转换为高精度
        p1 = to_decimal((x1, y1))
        p2 = to_decimal((x2, y2))
        # print(p1, p2)
        line = LineString([p1, p2])
        shapely_segments.append(line)
    for i in range(len(shapely_segments)):
        for j in range(i + 1, len(shapely_segments)):
            line1 = shapely_segments[i]
            line2 = shapely_segments[j]
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if intersection.is_empty:
                    continue
                if intersection.geom_type == 'Point':
                    intersection_points.add((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    for point in intersection:
                        intersection_points.add((point.x, point.y))
    return intersection_points

def find_intersections_and_split(segments):
    """
    根据高精度检测二维线段交点，并返回重新划分后的线段列表
    :param segments: 包含多个线段的列表，每个线段格式为 [(x1, y1, x2, y2), ...]
    :return: 划分后的线段列表
    """
    intersection_points = set()

    # 使用Shapely的LineString来表示线段
    shapely_segments = []
    for (x1, y1, x2, y2) in segments:
        # 使用Decimal转换为高精度
        p1 = to_decimal((x1, y1))
        p2 = to_decimal((x2, y2))
        # print(p1, p2)
        line = LineString([p1, p2])
        shapely_segments.append(line)

    # 计算交点
    for i in range(len(shapely_segments)):
        for j in range(i + 1, len(shapely_segments)):
            line1 = shapely_segments[i]
            line2 = shapely_segments[j]

            # 判断线段是否相交
            if line1.intersects(line2):
                # 获取交点
                intersection = line1.intersection(line2)
                if intersection.is_empty:
                    continue
                if intersection.geom_type == 'Point':
                    # 交点是一个点
                    intersection_points.add((intersection.x, intersection.y))
                elif intersection.geom_type == 'MultiPoint':
                    # 交点是多个点
                    for point in intersection:
                        intersection_points.add((point.x, point.y))

    # 将交点按坐标排序
    sorted_intersections = sorted(intersection_points, key=lambda p: (p[0], p[1]))
    print(len(sorted_intersections))
    # 根据交点重新划分线段
    split_segments = []
    intersection_points = [to_decimal(interpoint) for interpoint in intersection_points]
    new_segments = []
    interpoints_line_lst = [[] for i in intersection_points]
    interpoint_map = {}
    for i, interpoint in enumerate(intersection_points):
        interpoint_map[interpoint] = i

    for segment in segments:
        p1 = np.array((Decimal(segment[0]), Decimal(segment[1])))
        p2 = np.array((Decimal(segment[2]), Decimal(segment[3])))
        segment_points = [p1, p2]
        # print('p1, p2', p1, p2)
        for i, point in enumerate(intersection_points):
            point = np.array(point)
            if np.array_equal(point, p1) or np.array_equal(point, p2):
                continue
            # print(np.linalg.norm(p1 - point), np.linalg.norm(point - p2), np.linalg.norm(p1 - p2))
            # if np.linalg.norm(p1 - point) == np.NAN:
            #     continue
            d1 = p1 - point
            d2 = point - p2

            if (abs(np.linalg.norm(p1 - point) + np.linalg.norm(point - p2) - np.linalg.norm(p1 - p2)) < 0.0001) and abs(d1[1]*d2[0]-d1[0]*d2[1])<0.0001:
                segment_points.append(point)

        # 添加分割后的线段
        segment_points = sorted(segment_points, key=lambda p: np.linalg.norm(p - p1))
        for k in range(len(segment_points) - 1):
            if tuple(segment_points[k]) in intersection_points:
                idx = interpoint_map[tuple(segment_points[k])]
                interpoints_line_lst[idx].append(len(new_segments))
            if tuple(segment_points[k+1]) in intersection_points:
                idx = interpoint_map[tuple(segment_points[k+1])]
                interpoints_line_lst[idx].append(len(new_segments))
            new_segments.append((segment_points[k], segment_points[k + 1]))
    # print(interpoints_line_lst)
    return new_segments, intersection_points, interpoints_line_lst


def get_selected_points_by_range(point_cloud, x_min, x_max, y_min, y_max):
    filtered_points = point_cloud[
        (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
        (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max)
        ]
    return filtered_points

def get_selected_points_by_enlarge_range(points, x_range_min, x_range_max, y_range_min, y_range_max, gridsize, theta, mode):
    candidate_points = None
    if mode == 1: ## x equals
        for i in range(1, 2):
            selected_points1 = get_selected_points_by_range(points, x_range_min - gridsize * i, x_range_min - (i - 1) * gridsize, y_range_min, y_range_max)
            selected_points2 = get_selected_points_by_range(points, x_range_max + (i - 1) * gridsize, x_range_max + i * gridsize, y_range_min, y_range_max)
            if candidate_points is None:
                candidate_points = selected_points1 if selected_points1.shape[0] > selected_points2.shape[0] else selected_points2
            else:
                if candidate_points.shape[0]<selected_points1.shape[0]: candidate_points = selected_points1
                if candidate_points.shape[0]<selected_points2.shape[0]: candidate_points = selected_points2
            if selected_points1.shape[0]>theta or selected_points2.shape[0]>theta:
                return selected_points1 if selected_points1.shape[0]>selected_points2.shape[0] else selected_points2
    else:
        for i in range(1, 2):
            selected_points1 = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_min - gridsize * i, y_range_min - (i - 1) * gridsize)
            selected_points2 = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_max + (i - 1) * gridsize, y_range_max + gridsize * i)
            if candidate_points is None:
                candidate_points = selected_points1 if selected_points1.shape[0] > selected_points2.shape[0] else selected_points2
            else:
                if candidate_points.shape[0]<selected_points1.shape[0]: candidate_points = selected_points1
                if candidate_points.shape[0]<selected_points2.shape[0]: candidate_points = selected_points2
            if selected_points1.shape[0]>theta or selected_points2.shape[0]>theta:
                return selected_points1 if selected_points1.shape[0]>selected_points2.shape[0] else selected_points2
    return candidate_points

def get_points_by_roomline(roomline, points, theta = 20):
    gridsize = 0.09
    min_data = [-8.251500000000000, -26.517500000000000]
    height = 344
    width = 570
    st_x, st_y, ed_x, ed_y = roomline
    mode = 0
    ## calculate the region
    if st_x == ed_x:
        mode = 1
        x_range_min = min_data[0] + st_x * gridsize
        x_range_max = min_data[0] + (st_x + 1) * gridsize

        y_range_min = min_data[1] + st_y * gridsize
        y_range_max = min_data[1] + (ed_y + 1) * gridsize
        length = (ed_y - st_y)
        theta = theta * length
        selected_points = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_min, y_range_max)
        if selected_points.shape[0] < theta:
            selected_points1 = get_selected_points_by_enlarge_range(points, x_range_min, x_range_max, y_range_min, y_range_max, gridsize, theta, 1)
            # print('larger region',selected_points1.shape)
            if selected_points1.shape[0]>selected_points.shape[0]: selected_points = selected_points1
    else:
        mode = 2
        x_range_min = min_data[0] + st_x * gridsize
        x_range_max = min_data[0] + (ed_x + 1) * gridsize

        y_range_min = min_data[1] + st_y * gridsize
        y_range_max = min_data[1] + (st_y + 1) * gridsize
        length = (ed_x - st_x)
        theta = theta * length
        selected_points = get_selected_points_by_range(points, x_range_min, x_range_max, y_range_min, y_range_max)
        if selected_points.shape[0] < theta:
            selected_points1 = get_selected_points_by_enlarge_range(points, x_range_min, x_range_max, y_range_min, y_range_max, gridsize, theta, 2)
            # print('larger region',selected_points1.shape)
            if selected_points1.shape[0]>selected_points.shape[0]: selected_points = selected_points1
    room_range = {
        'x_min': x_range_min,
        'x_max': x_range_max ,
        'y_min': y_range_min ,
        'y_max': y_range_max ,
        'gridsize': gridsize
    }
    return selected_points, mode, theta, length, room_range

def distance(point, m, b):
    """计算点到直线的距离"""
    x, y = point
    return abs(-m * x + 1 * y - b) / np.sqrt(m ** 2 + 1)

def ransac_line_fitting(points, iterations=3000, threshold=0.05):
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
        if abs(min(m, 1/(m+1e-9))) > 1 / 20:  # 约束斜率
            continue
        if m==0:
            continue
        inliers = [point for point in points if distance(point, m, b) < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m = m
            best_b = b

    return best_m, best_b, np.array(best_inliers)

def project_point_on_line(x0, y0, k, b):
    # 计算直线的法向量
    # 法向量的分量
    A = -k
    B = 1
    C = -b
    # 计算投影点的 x 坐标
    # 使用公式：x' = (B(B*x0 - A*y0) - A*C) / (A^2 + B^2)
    x_prime = (B * (B * x0 - A * y0) - A * C) / (A ** 2 + B ** 2)
    # 计算投影点的 y 坐标
    y_prime = k * x_prime + b

    return x_prime, y_prime

def generate_virtual_points(roomline, num_points):
    gridsize = 0.09
    min_data = [-8.251500000000000, -26.517500000000000]

    st_x, st_y, ed_x, ed_y = roomline
    if st_x == ed_x:
        x_range_min = min_data[0] + st_x * gridsize
        x_range_max = min_data[0] + (st_x + 1) * gridsize
        y_range_min = min_data[1] + st_y * gridsize
        y_range_max = min_data[1] + (ed_y + 1) * gridsize
    else:
        x_range_min = min_data[0] + st_x * gridsize
        x_range_max = min_data[0] + (ed_x + 1) * gridsize
        y_range_min = min_data[1] + st_y * gridsize
        y_range_max = min_data[1] + (st_y + 1) * gridsize

    x_points = np.random.uniform(x_range_min, x_range_max, num_points)
    y_points = np.random.uniform(y_range_min, y_range_max, num_points)
    generated_virtual_points = np.column_stack((x_points, y_points))
    return generated_virtual_points

def get_ransac_line(selected_points, room_range, mode):
    if selected_points.shape[0]<2:
        return None, None, None
    m, b, _ = ransac_line_fitting(selected_points)
    print(m,b)
    if m ==None:
        return None, None, None
    if mode == 1:
        y_min = room_range['y_min']-room_range['gridsize']*2
        y_max = room_range['y_max']+room_range['gridsize']*2

        y_fit = np.linspace(y_min, y_max, 100)
        x_fit = (y_fit - b) / (m)

        start_point = np.array([x_fit[0], y_min])
        end_point = np.array([x_fit[-1], y_max])
        extension_length = 5
        direction = end_point - start_point
        direction_length = np.linalg.norm(direction)
        unit_direction = direction / direction_length
        new_start_point = start_point - unit_direction * extension_length
        new_end_point = end_point + unit_direction * extension_length
    else:
        x_min = room_range['x_min']-room_range['gridsize']*2
        x_max = room_range['x_max']+room_range['gridsize']*2

        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = m * x_fit + b

        start_point = np.array([x_min, y_fit[0]])
        end_point = np.array([x_max, y_fit[-1]])
        extension_length = 5
        direction = end_point - start_point
        direction_length = np.linalg.norm(direction)

        unit_direction = direction / direction_length
        new_start_point = start_point - unit_direction * extension_length
        new_end_point = end_point + unit_direction * extension_length
    return new_start_point, new_end_point, m

def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）

def point_to_line_segment_distance(px, py, ax, ay, bx, by):
    """
    计算点 (px, py) 到线段 (ax, ay) -> (bx, by) 的最短距离
    """
    # 计算向量 AB 和 AP
    ABx = bx - ax
    ABy = by - ay
    APx = px - ax
    APy = py - ay

    # 向量 AB 的平方长度
    AB_squared = ABx * ABx + ABy * ABy

    # 如果 A 和 B 重合（AB的长度为0），直接返回点到A的距离
    if AB_squared == 0:
        return math.sqrt(APx * APx + APy * APy)

    # 计算点 P 在 AB 向量上的投影系数 t
    t = (APx * ABx + APy * ABy) / AB_squared

    # 如果投影点在线段 AB 上，t 的值应该在 [0, 1] 范围内
    if t < 0:
        # 投影点在 A 之前，返回点 P 到 A 的距离
        closest_x = ax
        closest_y = ay
    elif t > 1:
        # 投影点在 B 之后，返回点 P 到 B 的距离
        closest_x = bx
        closest_y = by
    else:
        # 投影点在线段 AB 上，计算投影点的坐标
        closest_x = ax + t * ABx
        closest_y = ay + t * ABy

    # 计算点 P 到最近点 (closest_x, closest_y) 的距离
    dx = px - closest_x
    dy = py - closest_y
    return math.sqrt(dx * dx + dy * dy)

def calculate_supporting_points(lines, points):
    support_num = [0 for _ in lines]
    support_pp = [[] for _ in lines]
    for i, seg in enumerate(lines):
        for j, point in enumerate(points):
            point = np.array([Decimal(point[0]), Decimal(point[1])])
            if point_to_line_segment_distance(point[0], point[1], seg[0][0], seg[0][1], seg[1][0], seg[1][1]) < 0.05:
                support_num[i] += 1
                support_pp[i].append(point)
    return support_num, support_pp

def calculate_point_coverage(support_pp, lines):
    uncovered_length = [0 for _ in lines]
    for i, line in enumerate(lines):
        covered_length = 0
        segment_length = np.linalg.norm(line[0] - line[1])
        if len(support_pp[i]) == 0:
            uncovered_length[i] = float(segment_length)
            continue
        sorted_points = sorted(support_pp[i], key=lambda p: p[0])
        for j in range(1, len(sorted_points)):
            dist = np.linalg.norm(sorted_points[j] - sorted_points[j-1])
            if dist < 0.015:
                covered_length += dist
        uncovered_length[i] = 0 if float(segment_length - covered_length) < 0 else float(segment_length - covered_length)
    return uncovered_length

def save_list(out_path, data):
    with open(out_path, 'w') as file:
        for item in data:
            if isinstance(item, list):
                for i, ii in enumerate(item):
                    if i==0: file.write(f"{ii}")
                    else: file.write(f" {ii}")
                file.write(f"\n")
            else:
                file.write(f"{item}\n")

def save_line(out_path, data):
    with open(out_path, 'w') as file:
        for line in data:
            point1 = line[0]
            point2 = line[1]
            file.write(f"{float(point1[0])} {float(point1[1])} {float(point2[0])} {float(point2[1])}\n")

def save_result(output_path, support_num, uncovered_length, interpoint_list, seg_line, bbox_length, num_points):
    save_list(os.path.join(output_path, 'support_num.txt'), support_num)
    save_list(os.path.join(output_path, 'uncovered_length.txt'), uncovered_length)
    save_list(os.path.join(output_path, 'interpoint_list.txt'), interpoint_list)
    save_line(os.path.join(output_path, 'line.txt'), seg_line)
    with open(os.path.join(output_path, 'bbox_length.txt'), 'w') as file:
        file.write(f"{float(bbox_length)}")
    with open(os.path.join(output_path, 'num_points.txt'), 'w') as file:
        file.write(f"{int(num_points)}")

def get_bbox_length(points):
    points = np.vstack(points)
    x_min = np.min(points, axis=0)[0] - 1
    x_max = np.max(points, axis=0)[0] + 1
    y_min = np.min(points, axis=0)[1] - 1
    y_max = np.max(points, axis=0)[1] + 1
    # print(x_min, x_max, y_min, y_max)
    return math.sqrt((x_max - x_min)**2 + (y_max - y_min)**2), points.shape[0]

def get_interpoint_lst_by_resline(lines, interpoints):
    interpoint_list = [[] for _ in range(len(interpoints))]
    for i, point in enumerate(interpoints):
        for j, line in enumerate(lines):
            p1 = line[0]
            p2 = line[1]
            found1 = np.array_equal(p1, point)
            found2 = np.array_equal(p2, point)
            if found1==True or found2==True:
                interpoint_list[i].append(j)
    return interpoint_list

def calculate_dist_lines(line1, line2):
    intersect_points = find_intersection_between_lines([line1, line2])
    if intersect_points != set():
        dist_between_lines = 0
    else:
        p1 = (line1[0], line1[1])
        p2 = (line1[2], line1[3])
        p3 = (line2[0], line2[1])
        p4 = (line2[2], line2[3])
        dist1 = point_to_line_segment_distance(p1[0], p1[1], p3[0], p3[1], p4[0], p4[1])
        dist2 = point_to_line_segment_distance(p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])
        dist3 = point_to_line_segment_distance(p3[0], p3[1], p1[0], p1[1], p2[0], p2[1])
        dist4 = point_to_line_segment_distance(p4[0], p4[1], p1[0], p1[1], p2[0], p2[1])
        dist_between_lines = min(dist1, dist2, dist3, dist4)
    return dist_between_lines

def calculate_abs_slope(line1, line2):
    p1 = (line1[0], line1[1])
    p2 = (line1[2], line1[3])
    p3 = (line2[0], line2[1])
    p4 = (line2[2], line2[3])
    if p2[0] - p1[0] == 0:
        return None  # 垂直线，斜率无穷大
    else:
        slope1 = abs((p2[1] - p1[1]) / (p2[0] - p1[0]))
        theta_rad1 = math.atan(slope1)  # 获取角度（弧度）
        theta_deg1 = math.degrees(theta_rad1)  # 将弧度转换为角度
    if p4[0] - p3[0] == 0:
        return None  # 垂直线，斜率无穷大
    else:
        slope2 = abs((p4[1] - p3[1]) / (p4[0] - p3[0]))
        theta_rad2 = math.atan(slope2)  # 获取角度（弧度）
        theta_deg2 = math.degrees(theta_rad2)  # 将弧度转换为角度
    if slope1 == None and slope2 == None:
        return 0
    elif slope1 == None or slope2 == None:
        return 1
    else:
        return abs(theta_deg1 - theta_deg2)


def remove_similar_lines(lines, thres_dist, thres_angle, points):
    support_num = [0 for _ in lines]
    for i, seg in enumerate(lines):
        for j, point in enumerate(points):
            # point = np.array([Decimal(point[0]), Decimal(point[1])])
            if point_to_line_segment_distance(point[0], point[1], seg[0], seg[1], seg[2], seg[3]) < 0.05:
                support_num[i] += 1

    flag = [True for _ in lines]
    for i,line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j or flag[i] == False or flag[j] == False:
                continue
            # if i==1:
            #     print(calculate_dist_lines(line1, line2), calculate_abs_slope(line1, line2))
            if calculate_dist_lines(line1, line2) <thres_dist and calculate_abs_slope(line1, line2) < thres_angle:
                if support_num[i] > support_num[j]:
                    flag[j] = False
                else:
                    flag[i] = False
    res_lines = []
    print(flag)
    for idx, f in enumerate(flag):
        if f==True: res_lines.append(lines[idx])
    return res_lines

def define_prior_constraints(seg, floorplan, cur_idx):
    st_x, st_y, ed_x, ed_y = seg
    mode_xy = 0
    if st_x == ed_x:
        flag = 0
        for bias_x in [-1, 1, -2, 2]:
            for pos_y in range(st_y, ed_y+1):
                if st_x+bias_x >= 0 and st_x+bias_x< floorplan.shape[0]:
                    if floorplan[st_x+bias_x][pos_y]!=cur_idx and floorplan[st_x+bias_x][pos_y]!=0:
                        print(st_x, pos_y, bias_x)
                        flag = bias_x
                        mode_xy = 1
                        break
            if flag!=0: break
    else:
        flag = 0
        for bias_y in [-1, 1, -2, 2]:
            for pos_x in range(st_x, ed_x+1):
                if st_y+bias_y >= 0 and st_y+bias_y< floorplan.shape[1]:
                    if floorplan[pos_x][st_y+bias_y]!=cur_idx and floorplan[pos_x][st_y+bias_y]!=0:
                        print( pos_x, bias_y,st_y)
                        flag = bias_y
                        mode_xy = 2
                        break
            if flag!=0: break
    return flag, mode_xy

if __name__ == '__main__':

    area_name = r'E:\FloorSG\Expriment_output\S3DIS\Boundary_optim\area2.txt'
    img_orig = read_txt_to_matrix(area_name)
    image = np.array(img_orig,dtype=np.uint32)
    dir_path = r'E:\FloorSG\Expriment_output\S3DIS\Boundary_optim\room_boundary'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    point_path = r'E:\Stanford3dDataset_v1.2\Area_2\vertical_sub.txt'
    output_path = r'C:\2024\FloorSG\S3DIS_vector\area2'
    matrix = []
    with open(point_path, 'r') as file:
        for line in file:
            # 将每行的数字按空格或制表符分割，并转换为float类型，存储为一行矩阵
            row = list(map(float, line.split()))
            matrix.append(row)
    pointcloud = np.array(matrix)[:, 0:2]

    thres = 50

    fig = plt.figure(figsize=(10, 7))
    axes2 = fig.add_subplot(1, 1, 1)

    # plt.scatter(pointcloud[:, 0], pointcloud[:, 1], color='blue', label='Points', s=2)

    # np.savetxt('I:\\FloorSG\\Expriment_output\\S3DIS\\Boundary_optim\\area2_boundary.txt', boundary_map, fmt='%d', delimiter=' ')
    print(np.max(image))
    for i in range(42,43):
        print('第',i,'个房间')
        # if i!=5:9
        #     continue
        # if i!=1 and i!=17 and i!=18 and i!=19 and i!=28 and i!=31 and i!=34 and i!=37 and i!=41 and i!=42 and i!=43 and i!=44:
        #     continue7
        tmp_idx = np.argwhere(image==i)
        if tmp_idx.shape[0]==0:
            continue
        new_image = np.zeros(shape=image.shape, dtype=np.uint32)
        new_image[image == i] = 255
        cur_boundary_map = get_boundary_map(new_image)
        boundary_lst = []
        for x in range(cur_boundary_map.shape[0]):
            for y in range(cur_boundary_map.shape[1]):
                if cur_boundary_map[x][y]==1:
                    boundary_lst.append((x,y))
        roomline_path = os.path.join(dir_path, str(i) + '.txt')
        num_edge = 0
        cur_pos = boundary_lst[0]
        hash_map = np.zeros_like(cur_boundary_map)
        segments = []
        while 1:
            if num_edge == len(boundary_lst):
                break
            # print(cur_pos)
            for a, b in zip([1, 0, -1, 0], [0, 1, 0, -1]):
                if (cur_pos[0] + a < image.shape[0] and cur_pos[0] + a >= 0) and (cur_pos[1] + b < image.shape[1] and cur_pos[1] + b >= 0):
                    if cur_boundary_map[cur_pos[0] + a][cur_pos[1] + b]==1 and hash_map[cur_pos[0] + a][cur_pos[1] + b]==0:
                        res_st, res_ed = change_points(cur_pos, (cur_pos[0] + a, cur_pos[1] + b))
                        # with open(roomline_path, "a") as f:
                        #     f.write(f"{res_st[0]}\t{res_st[1]}\t{res_ed[0]}\t{res_ed[1]}\t{11}\t{11}\n")
                        segments.append((res_st[0],res_st[1],res_ed[0], res_ed[1]))
                        hash_map[cur_pos[0] + a][cur_pos[1] + b]=1
                        cur_pos=(cur_pos[0] + a,cur_pos[1] + b)
                        num_edge+=1
                        break

        merged_segments = merge_lines(segments)
        cur_room_lineseg = []
        cur_points = []
        buffer_points = []
        for num, seg in enumerate(merged_segments):

            # if num!=1 and num!=2 and num!=3 and num!=4 and num!=5:
            #     continue
            # print(seg)
            # print(num)
            selected_points, mode, theta, length, room_range = get_points_by_roomline(seg, pointcloud)
            cur_path = os.path.join(dir_path, str(i) + '_' + str(num) + '.txt')
            np.savetxt(cur_path, selected_points)
            selected_points = selected_points[:, 0:2]
            # print(length, selected_points.shape[0])
            if selected_points.shape[0]< theta/3 and length<3:
                continue
            print(num, seg, theta, length)
            print(selected_points.shape)
            _, _, k = get_ransac_line(selected_points, room_range, mode)
            if selected_points.shape[0] < theta/2 or k==None:
                # print(k, mode)
                if k == None:
                    selected_points = generate_virtual_points(seg, num_points=int(theta/2))
                # if mode == 1:
                #     if abs(k) < 1:
                #         continue
                # elif mode == 2:
                #     if abs(k) > 1:
                #         continue
                selected_points = generate_virtual_points(seg, num_points= int(theta/2))
            # plt.scatter(selected_points[:, 0], selected_points[:, 1], color='r', label='Points', s=2)

            print(selected_points.shape)
            bias_xy, mode_xy = define_prior_constraints(seg, image, i)
            # print('1111: ', bias_xy, mode_xy)

            #  fitting not vertical lines

            new_start_point, new_end_point, k = get_ransac_line(selected_points, room_range, mode)
            # print(new_start_point, new_end_point)
            if k==None:
                continue
            if mode == 1:
                if abs(k)<1.5:
                    continue
            elif mode ==2:
                if abs(k)>0.5:
                    continue
            alpha_xy = 0.03 * (3 - abs(bias_xy))
            if mode_xy == 1 and bias_xy != 0:
                new_start_point[0] += alpha_xy * (-bias_xy)
                new_end_point[0] += alpha_xy * (-bias_xy)
            elif mode_xy == 2 and bias_xy != 0:
                new_start_point[1] += alpha_xy * (-bias_xy)
                new_end_point[1] += alpha_xy * (-bias_xy)


            cur_points.append(selected_points)
            print('points.shape', (np.vstack(cur_points)).shape)
            cur_room_lineseg.append((new_start_point[0], new_start_point[1], new_end_point[0],new_end_point[1]))

            pointcloud = np.concatenate([pointcloud, selected_points], axis=0)
            #
            # plt.plot([new_start_point[0], new_end_point[0]], [new_start_point[1], new_end_point[1]], 'r--',
            #          label='Extended Line')

            # plt.plot(x_fit, y_fit, color='red', label='Fitted Line')
            # with open(roomline_path, "a") as f:
            #     res_st, res_ed = change_points((seg[0], seg[1]), (seg[2],seg[3]))
            #     f.write(f"{res_st[1]}\t{res_st[0]}\t{res_ed[1]}\t{res_ed[0]}\t{11}\t{11}\n")

        cur_path = os.path.join(output_path, str(i))
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        cur_room_lineseg = remove_similar_lines(cur_room_lineseg, thres_dist=0.02, thres_angle = 3, points=np.vstack(cur_points))
        for line in cur_room_lineseg:
            plt.plot([line[0], line[2]], [line[1], line[3]], 'r--', label='Extended Line')

        final_segments, inter_points, interpoints_line_lst = find_intersections_and_split(cur_room_lineseg)

        inter_points = [np.array(ii) for ii in inter_points]
        res_line = []
        for line in final_segments:
            p1 = line[0]
            p2 = line[1]
            found1 = any(np.array_equal(p1, arr) for arr in inter_points)
            found2 = any(np.array_equal(p2, arr) for arr in inter_points)

            if found1 == True and found2 == True:
                res_line.append(line)
        interpoints_line_lst = get_interpoint_lst_by_resline(res_line, inter_points)
        print(interpoints_line_lst)
        support_num, support_pp = calculate_supporting_points(res_line, np.vstack(cur_points))

        print(support_num)
        uncovered_length = calculate_point_coverage(support_pp, res_line)

        print(uncovered_length)
        bbox_length, num_points = get_bbox_length(cur_points)
        print(bbox_length, num_points)

        save_result(cur_path, support_num, uncovered_length, interpoints_line_lst, res_line, bbox_length, num_points)

        print(len(res_line))
        for segment in res_line:
            cur_line_color = random_color()
            plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color=cur_line_color, label='Split Segment')
        for point in inter_points:
            plt.scatter(point[0], point[1], color='black')  # 交点

        for p in cur_points:
            plt.scatter(p[:, 0], p[:, 1], color='blue', label='Points', s=2)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid()
    plt.show()