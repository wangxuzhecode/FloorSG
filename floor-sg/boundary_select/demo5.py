import matplotlib.pyplot as plt
from fractions import Fraction
from collections import defaultdict
import random

# 线段类表示
class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.p1 = (Fraction(x1), Fraction(y1))  # 第一个端点
        self.p2 = (Fraction(x2), Fraction(y2))  # 第二个端点

    def __repr__(self):
        return f"({self.p1}, {self.p2})"

    def get_points(self):
        return [self.p1, self.p2]

    def __eq__(self, other):
        return self.p1 == other.p1 and self.p2 == other.p2


# 计算两条线段的交点
def get_intersection(line1, line2):
    x1, y1 = line1.p1
    x2, y2 = line1.p2
    x3, y3 = line2.p1
    x4, y4 = line2.p2

    # 线段的参数化方程： P = P1 + t(P2 - P1) 和 Q = Q1 + u(Q2 - Q1)
    # 解线性方程组求交点

    # 计算分母
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行或重合的情况

    # 计算参数 t 和 u
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

    # 如果交点在线段内部（t 和 u 都在 [0, 1] 之间），则返回交点
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y)
    return None


# 划分线段
def split_segment(segment, intersection):
    x1, y1 = segment.p1
    x2, y2 = segment.p2
    xi, yi = intersection

    # 如果交点在端点之一，返回原线段
    if (xi == x1 and yi == y1) or (xi == x2 and yi == y2):
        return [segment]

    # 生成两个新线段
    new_segments = [
        LineSegment(x1, y1, xi, yi),
        LineSegment(xi, yi, x2, y2)
    ]
    return new_segments


# 检测所有线段交点并划分
def detect_and_split(segments):
    new_segments = []
    intersections = defaultdict(list)

    # 对每对线段进行交点检测
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            line1 = segments[i]
            line2 = segments[j]
            intersection = get_intersection(line1, line2)
            if intersection:
                intersections[intersection].append((line1, line2))

    # 根据交点划分线段
    for segment in segments:
        segments_to_process = [segment]
        # 先查找该线段与哪些交点相关
        for intersection in intersections:
            new_segment_list = []
            for s in segments_to_process:
                # 检查交点是否在线段内部，如果是则划分线段
                new_segments.extend(split_segment(s, intersection))
                new_segment_list.append(s)
            segments_to_process = new_segment_list
        # 去除重复线段
        new_segments = list(set(new_segments))
    return new_segments


# 可视化函数
def visualize_segments(segments):
    plt.figure(figsize=(8, 8))

    for i, segment in enumerate(segments):
        x_values = [float(segment.p1[0]), float(segment.p2[0])]
        y_values = [float(segment.p1[1]), float(segment.p2[1])]
        plt.plot(x_values, y_values, marker='o', label=f"Segment {i+1}")

    plt.title("Line Segments and their Intersections")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()


# 测试数据
segments = [
    LineSegment(0, 0, 2, 2),
    LineSegment(0, 2, 2, 0),
    LineSegment(1, 0, 1, 3),
    LineSegment(2, 0, 2, 3)
]

# 获取交点并划分线段
new_segments = detect_and_split(segments)

# 可视化结果
visualize_segments(new_segments)
