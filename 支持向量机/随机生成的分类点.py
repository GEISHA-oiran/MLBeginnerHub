import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 创建50个数据点，并将它们分为2类
x, y = make_blobs(n_samples=30, centers=2, random_state=6)
print(y)  # 打印分类标签

# 构建一个内核为线性的支持向量机模型
clf = svm.SVC(kernel="linear", C=100)  # C越大惩罚力度越高，泛化能力越弱
clf.fit(x, y)  # 拟合模型

# 绘制散点图
# x的第一列作为x坐标，x的第二列作为y坐标，c=y表示根据y的值对散点进行着色，s=30表示散点的大小为30，cmap="Paired"表示使用"Paired"颜色映射。
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="Set2", alpha=0)

# 建立图形坐标
ax = plt.gca()
xlim = ax.get_xlim()  # 获取数据点x坐标的最大值和最小值
ylim = ax.get_ylim()  # 获取数据点y坐标的最大值和最小值

# 根据坐标轴生成等差数列(这里是对参数进行网格搜索)
xx = np.linspace(xlim[0], xlim[1], 30)  # 在x轴的范围内生成30个等距的点
yy = np.linspace(ylim[0], ylim[1], 30)  # 在y轴的范围内生成30个等距的点
YY, XX = np.meshgrid(yy, xx)  # 创建网格点坐标矩阵
xy = np.vstack([XX.ravel(), YY.ravel()]).T  # 将网格点坐标展开成一维数组，并转置
Z = clf.decision_function(xy).reshape(XX.shape)  # 计算每个网格点到分类边界的距离，并将结果变形成与网格点坐标相同的形状

# x的第一列作为x坐标，x的第二列作为y坐标，c=y表示根据y的值对散点进行着色，s=30表示散点的大小为30，cmap="Paired"表示使用"Paired"颜色映射。
plt.contourf(XX, YY, Z, cmap="RdPu", alpha=0.5)  # 用彩色区域表示不同的决策区域
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap="rainbow", alpha=0.25)# 标记支持向量
ax.contour(XX, YY, Z, colors='white', levels=[-1, 0, 1], alpha=1, linestyles=["--", "-", "--"])  # 绘制分类的边界
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors="none")
plt.show()  # 显示图形
