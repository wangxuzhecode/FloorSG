import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, Matrix
from fractions import Fraction


def line_intersection(P1, P2, Q1, Q2):
    """
    计算两条线段P1P2与Q1Q2的交点
    返回交点坐标 (x, y)，如果无交点返回None
    """
    x1, y1 = P1
    x2, y2 = P2
    x3, y3 = Q1
    x4, y4 = Q2

    # 使用符号变量计算交点
    t, u = symbols('t u')

    # 计算方向向量
    A = Matrix([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]])
    b = Matrix([[x3 - x1], [y3 - y1]])

    # 解方程组
    try:
        sol = A.LUsolve(b)
    except Exception as e:
        return None  # 如果无法解方程，则说明线段不相交

    t_val, u_val = sol[0], sol[1]

    # 判断交点是否在两个线段的范围内
    if 0 <= t_val <= 1 and 0 <= u_val <= 1:
        # 计算交点坐标
        intersection_x = x1 + t_val * (x2 - x1)
        intersection_y = y1 + t_val * (y2 - y1)
        return Fraction(intersection_x), Fraction(intersection_y)
    else:
        return None  # 没有交点


def divide_segment(P1, P2, intersection):
    """根据交点将线段P1P2划分为两个子线段"""
    x1, y1 = P1
    x2, y2 = P2
    x_int, y_int = intersection

    # 创建两个子线段
    sub_segment1 = [(x1, y1), (x_int, y_int)]
    sub_segment2 = [(x_int, y_int), (x2, y2)]

    return sub_segment1, sub_segment2


def check_and_divide(segments):
    """检查并划分多条线段，直到没有交点"""
    new_segments = []
    intersections = []

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg1 = segments[i]
            seg2 = segments[j]

            # 计算交点
            intersection = line_intersection(seg1[0], seg1[1], seg2[0], seg2[1])

            if intersection:
                intersections.append(intersection)
                sub_segments1, sub_segments2 = divide_segment(seg1[0], seg1[1], intersection)
                new_segments.extend([sub_segments1, sub_segments2])
                sub_segments1, sub_segments2 = divide_segment(seg2[0], seg2[1], intersection)
                new_segments.extend([sub_segments1, sub_segments2])
            else:
                # 如果没有交点，直接添加原线段
                new_segments.append(seg1)
                new_segments.append(seg2)

    # 去重，避免重复的子线段
    new_segments = list({tuple(segment) for segment in new_segments})

    return new_segments, intersections

def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）

def plot_segments(segments, intersections):
    """可视化所有线段和交点"""
    plt.figure(figsize=(8, 8))

    # 绘制线段
    for segment in segments:
        x_values = [point[0] for point in segment]
        y_values = [point[1] for point in segment]
        cur_color = random_color()
        plt.plot(x_values, y_values, marker='o', linestyle='--', color=cur_color, alpha=1)

    # 绘制交点
    if intersections:
        x_int = [point[0] for point in intersections]
        y_int = [point[1] for point in intersections]
        # plt.scatter(x_int, y_int, color='red', zorder=5, label='Intersection Points')

    # 添加标签和标题
    plt.title('Line Segments and Intersection Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.gca().set_aspect('equal', adjustable='box')

    # 显示图像
    plt.show()


segments1 = [(-3.521951711789071, -0.6570867444295635, 0.6759517117890705, -0.5000613518167314), (-1.587805234737911, -2.6162792747064554, -1.3490679695456596, 5.082279274706456), (-3.5229999999999997, 3.23099995, 3.777000000000001, 3.23099995), (1.73275428812203, -3.6169999119705905, 1.7357348800861128, 5.082999911970591), (-0.8222976709354657, -1.6363351395255288, 3.7762976709354663, -1.495562765586934), (1.289720835600083, -3.916328145705081, 1.4214169235594474, 0.4823281457050814), (-3.4229841315535086, -1.966237306029704, 3.2769841315535113, -1.9354187160081626), (-1.5001958865245149, -3.9157654635109505, -1.2851342038309606, 1.3817654635109506)]

# 示例
# segments = [
#     [(Fraction(0), Fraction(0)), (Fraction(2), Fraction(2))],
#     [(Fraction(0), Fraction(2)), (Fraction(2), Fraction(0))],
#     [(Fraction(1), Fraction(0)), (Fraction(1), Fraction(3))]
# ]
segments = []
for seg in segments1:
    segments.append([(Fraction(seg[0]), Fraction(seg[1])), (Fraction(seg[2]), Fraction(seg[3]))])

# 执行交点检测和线段划分
new_segments, intersections = check_and_divide(segments)
print(len(new_segments))
# 输出划分后的线段和交点
# print(f"交点: {intersections}")
# print(f"划分后的线段:")
for segment in new_segments:
    print(segment)

# 绘制线段和交点
plot_segments(new_segments, intersections)
