import numpy as np
import matplotlib.pyplot as plt


def distance(point, m, b):
    """计算点到直线的距离"""
    x, y = point
    return abs(-m * x + 1 * y - b) / np.sqrt(m ** 2 + 1)

def ransac_line_fitting(points, iterations=1000, threshold=0.05):
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
        # if abs(min(m, 1/(m+1e-9))) > 1 / 20:  # 约束斜率
        #     continue
        if m==0:
            continue
        inliers = [point for point in points if distance(point, m, b) < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m = m
            best_b = b

    return best_m, best_b, np.array(best_inliers)

def load_point_cloud(file_path):
    points = np.loadtxt(file_path)
    return points

file_path = r'E:\Stanford3dDataset_v1.2\area1_door\door_1.txt'  # 修改为你的点云文件路径
points = load_point_cloud(file_path)
print(points.shape)
idx = np.argwhere((points[:,2]>1.8))
points = points[idx]
points = points[:,0,0:2]
print(points.shape)

m, b, _ = ransac_line_fitting(points,threshold=0.03)
x_fit = np.linspace(np.min(points,0)[0], np.max(points,0)[0], 100)
y_fit = m * x_fit + b
start_point = np.array([np.min(points,0)[0], y_fit[0]])
end_point = np.array([np.max(points,0)[0], y_fit[-1]])
extension_length = 0
direction = end_point - start_point
direction_length = np.linalg.norm(direction)

unit_direction = direction / direction_length
new_start_point = start_point - unit_direction * extension_length
new_end_point = end_point + unit_direction * extension_length

plt.figure(figsize=(8, 6))

plt.plot([new_start_point[0], new_end_point[0]], [new_start_point[1], new_end_point[1]], 'r--', label='Extended Line')

plt.scatter(points[:, 0], points[:, 1], color='blue', label='Point Cloud')

inliers = np.array([point for point in points if distance(point, m, b) > 0.08])
plt.scatter(inliers[:, 0], inliers[:, 1], color='red', label='Point Cloud')

m, b, _ = ransac_line_fitting(inliers)
x_fit = np.linspace(np.min(inliers,0)[0], np.max(inliers,0)[0], 100)
y_fit = m * x_fit + b
start_point = np.array([np.min(inliers,0)[0], y_fit[0]])
end_point = np.array([np.max(inliers,0)[0], y_fit[-1]])
y_min = np.min(inliers, 0)[1]
y_max = np.max(inliers, 0)[1]
new_x_min = (y_min-b)/m
new_x_max = (y_max-b)/m


plt.plot([new_x_min, new_x_max], [y_min, y_max], 'r--', label='Extended Line')
print(new_start_point, new_end_point)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC Two Line Fitting')
plt.legend()
plt.grid(True)
plt.show()
