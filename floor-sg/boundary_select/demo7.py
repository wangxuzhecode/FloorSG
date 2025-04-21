import math


# 计算两点间的距离
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# 计算两点间的斜率，返回None表示无穷大（垂直线）
def slope(p1, p2):
    if p2[0] - p1[0] == 0:
        return None  # 垂直线，斜率无穷大
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


# 判断两条线段是否距离过近且斜率相近
def is_similar(line1, line2, distance_threshold, slope_threshold):
    # 取两条线段的端点
    p1, p2 = line1
    q1, q2 = line2

    # 计算线段端点之间的距离
    dist = min(distance(p1, q1), distance(p1, q2), distance(p2, q1), distance(p2, q2))

    if dist > distance_threshold:
        return False  # 如果距离超过阈值，则认为不相似

    # 计算两条线段的斜率
    slope1 = slope(p1, p2)
    slope2 = slope(q1, q2)

    if slope1 is None and slope2 is None:
        return True  # 两条都是垂直线，斜率相同
    elif slope1 is None or slope2 is None:
        return False  # 只有一个是垂直线，斜率不同
    else:
        return abs(slope1 - slope2) < slope_threshold  # 判断斜率差值是否小于阈值


# 删除相似的线段
def remove_similar_lines(lines, distance_threshold, slope_threshold):
    result = []
    for i, line1 in enumerate(lines):
        keep = True
        for j, line2 in enumerate(lines):
            if i != j and is_similar(line1, line2, distance_threshold, slope_threshold):
                keep = False
                break
        if keep:
            result.append(line1)
    return result


# 示例线段
lines = [
    [(0, 0), (2, 2)],  # 斜率为 1
    [(1, 1), (3, 3)],  # 斜率为 1
    [(0, 0), (1, 0)],  # 斜率为 0
    [(5, 5), (6, 7)]  # 斜率为 2
]

# 设置距离和斜率的阈值
distance_threshold = 2.0  # 距离阈值
slope_threshold = 0.1  # 斜率阈值

# 删除相似的线段
filtered_lines = remove_similar_lines(lines, distance_threshold, slope_threshold)

# 打印结果
print("剩余的线段:")
for line in filtered_lines:
    print(line)
