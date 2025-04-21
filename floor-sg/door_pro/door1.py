import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

def load_point_cloud(file_path):
    points = np.loadtxt(file_path)
    return points


def distance_point_to_segment(px, py, x1, y1, x2, y2):
    # 向量 AB 和 AP
    AB = np.array([x2 - x1, y2 - y1])
    AP = np.array([px - x1, py - y1])

    # 向量 AB 的平方
    AB_sq = AB.dot(AB)

    # 向量 AP 与 AB 的点积
    projection = AP.dot(AB) / AB_sq

    # 判断投影位置
    if projection < 0:
        # 点 P 离 A 点最近
        closest_x, closest_y = x1, y1
    elif projection > 1:
        # 点 P 离 B 点最近
        closest_x, closest_y = x2, y2
    else:
        # 点 P 在线段的投影点
        closest_x = x1 + projection * AB[0]
        closest_y = y1 + projection * AB[1]

    # 计算 P 点与最近点的距离
    distance = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    return distance, projection


def find_points_near_segment(points, x1, y1, x2, y2, epsilon):
    near_points = []
    for px, py in points:
        distance, projection = distance_point_to_segment(px, py, x1, y1, x2, y2)
        if distance <= epsilon and 0 <= projection <= 1:
            continue
        else:
            near_points.append((px, py))
    return np.array(near_points)

def ransac_fit_two_lines(points):
    # 拟合第一条直线
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]

    ransac1 = RANSACRegressor(residual_threshold=0.05)
    ransac1.fit(X, y)

    # 获取第一条拟合直线
    line_x = np.linspace(np.min(X), np.max(X), 100)
    line_y = ransac1.predict(line_x.reshape(-1, 1))

    # 去除第一条直线的点
    inlier_mask = ransac1.inlier_mask_
    points_remaining = points[~inlier_mask]  # 剩余点
    print(points.shape)
    points= points[:, 0:2]
    x1,y1,x2,y2 = line_x[0], line_y[0], line_x[-1], line_y[-1]
    points_remaining = find_points_near_segment(points, x1, y1, x2, y2, 0.1)
    print(points_remaining.shape)
    # 拟合第二条直线
    X_remaining = points_remaining[:, 0].reshape(-1, 1)
    y_remaining = points_remaining[:, 1]

    ransac2 = RANSACRegressor(residual_threshold=0.1, max_trials=2000000,loss = "squared_error")
    ransac2.fit(X_remaining, y_remaining)
    inlier_mask = ransac2.inlier_mask_
    # 获取第二条拟合直线
    line_x2 = np.linspace(np.min(X_remaining), np.max(X_remaining), 100)
    line_y2 = ransac2.predict(line_x2.reshape(-1, 1))

    return line_x, line_y, line_x2, line_y2, points_remaining[inlier_mask]


# 可视化两条拟合直线
def visualize_two_lines(points, line_x, line_y, line_x2, line_y2, inner_p):
    plt.figure(figsize=(8, 6))

    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Point Cloud')
    plt.scatter(inner_p[:, 0], inner_p[:, 1], color='r', label='Point Cloud')

    plt.plot(line_x, line_y, color='red', label='Fitted Line 1', linewidth=2)
    plt.plot(line_x2, line_y2, color='green', label='Fitted Line 2', linewidth=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RANSAC Two Line Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()


# 主程序
if __name__ == "__main__":
    file_path = r'E:\Stanford3dDataset_v1.2\area1_door\door_1.txt'  # 修改为你的点云文件路径
    points = load_point_cloud(file_path)
    print(points.shape)
    idx = np.argwhere((points[:,2]>2.02))
    points = points[idx]
    points = points[:,0,:]
    print(points.shape)
    # 拟合两条直线
    line_x, line_y, line_x2, line_y2, inner_p = ransac_fit_two_lines(points)

    # 可视化两条直线
    visualize_two_lines(points, line_x, line_y, line_x2, line_y2, inner_p)
