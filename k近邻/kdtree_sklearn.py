import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree
np.random.seed(0)
# 随机产生150个二维数据
points = np.random.random((150, 2))
tree = KDTree(points)
point = points[0]
# k近邻发搜索
dists, indices = tree.query([point], k=4)

# q指定半径搜索
indices = tree.query_radius([point], r=0.2)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.add_patch(Circle(point, 0.2, color='g', fill=False))
X, Y = [p[0] for p in points], [p[1] for p in points]
plt.scatter(X, Y)
plt.scatter([point[0]], [point[1]], c='r')
plt.show()