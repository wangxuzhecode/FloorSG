import numpy as np
import matplotlib.pyplot as plt


def intersect(p1, p2, p3, p4):
    """ 检测线段 p1p2 和 p3p4 是否相交，并返回交点 """
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    determinant = A1 * B2 - A2 * B1

    if determinant == 0:
        return None  # 平行或重合的线段

    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    # 检查交点是否在两个线段上
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
            min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]) and
            min(p3[0], p4[0]) <= x <= max(p3[0], p4[0]) and
            min(p3[1], p4[1]) <= y <= max(p3[1], p4[1])):
        return np.array([x, y])

    return None


def split_segments(segments):
    """ 根据交点分割线段 """
    points = []  # 保存交点
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            p1, p2 = segments[i]
            p3, p4 = segments[j]
            intersection = intersect(p1, p2, p3, p4)
            if intersection is not None:
                points.append(intersection)

    # 去重并排序交点
    points = np.unique(points, axis=0)

    # 根据交点分割线段
    new_segments = []
    for segment in segments:
        p1, p2 = segment
        print(p1)
        segment_points = [p1, p2]
        for point in points:
            if np.array_equal(point, p1) or np.array_equal(point, p2):
                continue
            if np.linalg.norm(p1 - point) + np.linalg.norm(point - p2) == np.linalg.norm(p1 - p2):
                segment_points.append(point)

        # 添加分割后的线段
        segment_points = sorted(segment_points, key=lambda p: np.linalg.norm(p - p1))
        for k in range(len(segment_points) - 1):
            new_segments.append((segment_points[k], segment_points[k + 1]))

    return new_segments
def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）
# 示例线段
segments = [
    (np.array([1, 1]), np.array([5, 4])),
    (np.array([2, 4]), np.array([6, 1])),
    (np.array([4, 0]), np.array([4, 5])),
]
# 分割线段
final_segments = split_segments(segments)

# 可视化结果
plt.figure(figsize=(8, 6))
for segment in segments:
    plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'b-', label='Original Segment')
for segment in final_segments:
    plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r--', label='Split Segment')

# 提取并绘制交点
unique_segments = np.unique(final_segments, axis=0)
points = []
for segment in unique_segments:
    points.append(segment[0])
    points.append(segment[1])
points = np.unique(points, axis=0)

plt.scatter(points[:, 0], points[:, 1], color='black')  # 交点
plt.title('Segment Splitting by Intersections')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.axis('equal')
plt.legend()
plt.show()
