import numpy as np
from shapely.geometry import LineString
from decimal import Decimal, getcontext

# 设置高精度
getcontext().prec = 50  # 设置精度为50位


def to_decimal(point):
    """将普通浮点数转换为高精度Decimal"""
    return (Decimal(point[0]), Decimal(point[1]))


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
            if abs(np.linalg.norm(p1 - point) + np.linalg.norm(point - p2) - np.linalg.norm(p1 - p2)) < 0.0001:
                segment_points.append(point)

        # 添加分割后的线段
        segment_points = sorted(segment_points, key=lambda p: np.linalg.norm(p - p1))
        for k in range(len(segment_points) - 1):
            if tuple(segment_points[k]) in intersection_points:
                idx = interpoint_map[tuple(segment_points[k])]
                print(idx)
                interpoints_line_lst[idx].append(len(new_segments))
            if tuple(segment_points[k+1]) in intersection_points:
                idx = interpoint_map[tuple(segment_points[k+1])]
                print(idx)
                interpoints_line_lst[idx].append(len(new_segments))
            new_segments.append((segment_points[k], segment_points[k + 1]))
    print(interpoints_line_lst)
    return new_segments, intersection_points

def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）

# # 示例使用
# segments = [
#     (0, 0, 5, 5),  # 线段1
#     (0, 5, 5, 0),  # 线段2
#     (1, 1, 4, 1),  # 线段3
#     (2, 2, 3, 2)  # 线段4
# ]
segments = [(-3.521951711789071, -0.6570867444295635, 0.6759517117890705, -0.5000613518167314), (-1.587805234737911, -2.6162792747064554, -1.3490679695456596, 5.082279274706456), (-3.5229999999999997, 3.23099995, 3.777000000000001, 3.23099995), (1.73275428812203, -3.6169999119705905, 1.7357348800861128, 5.082999911970591), (-0.8222976709354657, -1.6363351395255288, 3.7762976709354663, -1.495562765586934), (1.289720835600083, -3.916328145705081, 1.4214169235594474, 0.4823281457050814), (-3.4229841315535086, -1.966237306029704, 3.2769841315535113, -1.9354187160081626), (-1.5001958865245149, -3.9157654635109505, -1.2851342038309606, 1.3817654635109506)]
split_segments, inter_points = find_intersections_and_split(segments)
print('111111', len(segments))
print(len(split_segments))
# 输出划分后的线段
# for segment in split_segments:
#     print(f"Segment: ({segment[0][0]}, {segment[0][1]}) -> ({segment[1][0]}, {segment[1][1]})")
import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
for segment in split_segments:
    print(segment[0])
    cur_color = random_color()
    plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], color=cur_color, label='Original Segment')
print(inter_points)
for point in inter_points:
    plt.scatter(point[0], point[1], color='black')  # 交点

# for segment in split_segments:
#     print(segment)
#     plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r--', label='Split Segment')

# plt.scatter(points[:, 0], points[:, 1], color='black')  # 交点
plt.title('Segment Splitting by Intersections')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()