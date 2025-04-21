import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 生成一些示例数据（二维点集）
points = np.loadtxt('./s3dis_area2_room19.txt')

from sklearn.cluster import KMeans

# 生成一些示例数据（二维点集）

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=15, random_state=0).fit(points)

# 获取聚类标签
labels = kmeans.labels_

# 可视化聚类结果
plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# 打印聚类标签
print("Cluster labels:", labels)

