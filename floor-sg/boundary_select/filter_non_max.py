import numpy as np
from collections import defaultdict


# 计算多边形的面积（使用高斯面积公式）
def polygon_area(vertices):
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# 通过线段构建多边形的顶点
def build_polygons(segments):
    # 记录端点连接关系
    adj = defaultdict(list)

    for (x1, y1), (x2, y2) in segments:
        adj[(x1, y1)].append((x2, y2))
        adj[(x2, y2)].append((x1, y1))

    polygons = []
    visited = set()

    # 遍历图，提取每个多边形
    for start in adj:
        if start not in visited:
            polygon = []
            current = start
            prev = None
            while current not in visited:
                polygon.append(current)
                visited.add(current)
                # 获取当前点的下一个点
                next_point = adj[current]
                next_point.remove(prev)  # 避免回到上一个点
                prev = current
                current = next_point[0]  # 选择一个相邻点
            polygons.append(polygon)

    return polygons


# 线段数据（每条线段由两个端点组成）
segments = [
    ((0, 0), (4, 0)),
    ((4, 0), (4, 3)),
    ((4, 3), (0, 3)),
    ((0, 3), (0, 0)),
    ((2, 1), (5, 1)),
    ((5, 1), (5, 4)),
    ((5, 4), (2, 4)),
    ((2, 4), (2, 1))
]

# 构建多边形
polygons = build_polygons(segments)

# 计算每个多边形的面积
areas = []
for polygon in polygons:
    # 确保顶点按顺序排列
    polygon = np.array(polygon)
    area = polygon_area(polygon)
    areas.append(area)

# 找到最大面积的多边形
max_area_index = np.argmax(areas)
max_area_polygon = polygons[max_area_index]
max_area = areas[max_area_index]

print(f"面积最大的多边形是: {max_area_polygon}")
print(f"该多边形的面积为: {max_area}")
