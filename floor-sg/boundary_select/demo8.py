import numpy as np
import matplotlib.pyplot as plt
from jedi.api.refactoring import inline
from sklearn.linear_model import RANSACRegressor
from sympy import print_tree

# 生成一些模拟的边界点云（带有噪声）
np.random.seed(42)
n_points = 300
x = np.linspace(-10, 10, n_points)
y = 2 * x + np.random.normal(scale=3, size=n_points)  # 加入一些噪声

points = np.loadtxt('./s3dis_area2_room19.txt')
x = points[:,0]
y = points[:,1]

# 设定一个阈值，当点的误差小于该阈值时认为它属于当前拟合的直线
distance_threshold = 0.05
inner_points= []

# 定义拟合函数，RANSAC 拟合多条线段
def fit_multiple_lines(x, y, distance_threshold=0.01):
    # 数据集的点集
    points = np.column_stack((x, y))

    # 存储拟合的直线参数
    lines = []

    while len(points) > 0:
        # 创建 RANSAC 模型

        ransac = RANSACRegressor(residual_threshold=distance_threshold)

        # 拟合直线
        ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

        # 获取当前拟合直线的模型参数
        inlier_mask = ransac.inlier_mask_

        # 提取内点
        inlier_points = points[inlier_mask]
        if inlier_points.shape[0]<100:
            continue
        inner_points.append(inlier_points)
        # 存储拟合直线的参数
        lines.append((ransac.estimator_.coef_[0], ransac.estimator_.intercept_))

        # 剔除已拟合的点
        points = points[~inlier_mask]
        if len(points)<1000:
            break

    return lines

def random_color():
    return np.random.rand(3,)  # 生成 RGB 颜色值（0-1之间的随机数）

# 拟合多条线段
lines = fit_multiple_lines(x, y, distance_threshold)

for p in inner_points:
# 绘图展示拟合结果
    print(p.shape)
    plt.scatter(p[:, 0], p[:,1], color=random_color(), alpha=0.5)


#
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("RANSAC 多条线段拟合")
plt.grid(True)
plt.show()
