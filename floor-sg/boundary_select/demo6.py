import math


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


# 示例
px, py = 3, 4  # 点 P 的坐标
ax, ay = 1, 1  # 线段 A 的坐标
bx, by = 5, 1  # 线段 B 的坐标

distance = point_to_line_segment_distance(px, py, ax, ay, bx, by)
print(f"点到线段的最短距离是: {distance}")
