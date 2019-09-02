from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# 构建正态分布来产生数字,20行2列*2
train_x = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]

# 构建 20个class0，20个class1
label_y = [0] * 20 + [1] * 20

# svm设置：
clf = svm.SVC(kernel='linear')
clf.fit(train_x, label_y)

# 获取weights
weights = clf.coef_[0]

rate = -weights[0] / weights[1]  # 斜率
# 画图划线
xx = np.linspace(-5, 5)  # (-5,5)之间x的值
yy = rate * xx - (clf.intercept_[0]) /weights[1]  # xx带入y，截距

# 画出与点相切的线
b = clf.support_vectors_[0]
#yy_down 是线 yy的下边的边界线
yy_down = rate * xx + (b[1] - rate* b[0])
b = clf.support_vectors_[-1]
#yy_up 是线 yy的上边的边界线
yy_up =rate * xx + (b[1] - rate * b[0])


# 测试, 两个中括号！
for i in range(20):
    test_x = np.random.randn(1, 2) * 10
    print('测试:点({},{}) 所属类别{}'.format(test_x[0][0],test_x[0][1], clf.predict(test_x)))

plt.figure(figsize=(8, 4))
plt.plot(xx, yy)
#绘制左上角的线
plt.plot(xx, yy_up)
#绘制 左下角的线
plt.plot(xx, yy_down)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
plt.scatter(train_x[:, 0], train_x[:, 1], c=label_y, cmap=plt.cm.Paired)  # [:，0]列切片，第0列
plt.axis('tight')
plt.show()