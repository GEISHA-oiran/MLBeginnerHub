import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 第一步：加载数据
# area 是商品房面积
area = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00,
                 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])  # (16, )

# room 是商品房房间数
room = np.array([3, 2, 2, 3, 1, 2, 3, 2,
                 2, 3, 1, 1, 1, 1, 2, 2])

# price 是样本房价
price = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00,
                  62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

# 第二步：数据处理
num = len(area)

# 创建元素值全为1的一维数组 x0
x0 = np.ones(num)
# x1 是商品房面积归一化后的结果
x1 = (area - area.min()) / (area.max() - area.min())
# x2 是商品房房间数归一化后的结果
x2 = (room - room.min()) / (room.max() - room.min())

# 将 x0、x1、x2堆叠为形状为 (16, 3) 的二维数组
X = np.stack((x0, x1, x2), axis=1)

# 将 price 转换为形状为 (16, 1) 的二维数组
Y = price.reshape(-1, 1)

# 第三步：设置超参数 学习率，迭代次数
learn_rate = 0.0001
itar = 1000000  # 迭代次数为1000000次

display_step = 50000  # 每循环50000次显示一次训练结果

# 第四步：设置模型参数的初始值
np.random.seed(612)
W = np.random.randn(3, 1)

# 第五步：训练模型 W
mse = []  # 这是个Python列表, 用来保存每次迭代后的损失值

# 下面使用 for 循环来实现迭代
# 循环变量从 0 开始, 到 101 结束,循环 101 次, 为了描述方便, 以后就说迭代 100 次
# 同样, 当 i 等于 10 时, 我们就说第十次迭代
for i in range(0, itar + 1):
    # 首先计算损失函数对 W 的偏导数
    dL_dW = np.matmul(np.transpose(X), np.matmul(X, W)-Y)
    # 然后使用迭代公式更新 W
    W = W - learn_rate*dL_dW

    # 我们希望能够观察到每次迭代的结果, 判断是否收敛或者什么时候开始收敛
    # 因此需要使用每次迭代后的 W 来计算损失, 并且把它显示出来

    # 这里的 X 形状为 (16, 3), W 形状为 (3, 1), 得到 Y_PRED 的形状为 (16, 1)
    Y_PRED = np.matmul(X, W)  # 使用当前这次循环得到的W, 计算所有样本的房价的估计值
    Loss = np.mean(np.square(Y - Y_PRED)) / 2  # 使用房价的估计值和实际值计算均方误差
    mse.append(Loss)  # 把得到的均方误差加入列表 mse

    if i % display_step == 0:
        print("i:%i, Loss：%f" % (i, mse[i]))
        """
        i:0, Loss:4368.213908
        i:500000, Loss：79.871073
        i:1000000, Loss：79.871073
        """
print(W)
"""
[[51.39029673]
[48.74950958]
[28.66300756]]
"""

# 第六步：样本数据可视化

# 创建Figure对象
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(range(0, 5000), mse[0:5000])
plt.xlabel('Iteration', color='r', fontsize=14)
plt.ylabel('Loss', color='r', fontsize=14)
plt.title("前5000次迭代的损失值变化曲线图", fontsize=14)

plt.subplot(1, 2, 2)
Y_PRED = Y_PRED.reshape(-1)
plt.plot(price, color="red", marker='o', label="销售记录")
plt.plot(Y_PRED, color="blue", marker='.', label="预测房价")
plt.xlabel('Sample', color='r', fontsize=14)
plt.ylabel('Price', color='r', fontsize=14)
plt.title("估计值 & 标签值", fontsize=14)
plt.legend(loc="upper right")

plt.suptitle("梯度下降法求解多元线性回归", fontsize=18)

# 将创建好的图像显示出来
plt.show()
