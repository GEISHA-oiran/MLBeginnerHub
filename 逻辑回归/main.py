import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing as pp


# 读取并查看数据
path = r"C:\Users\29293\Desktop\-\ML机器学习\机器学习数据集\逻辑回归.txt"
pdData = pd.read_csv(path, header=None, names=['exam_1', 'exam_2', 'admitted'])
pdData.head()
pdData.shape

# 根据 admitted 画出数据图像
# 返回行的子集，例如 Admit = 1，即 *positive* 示例集
positive = pdData[pdData['admitted'] == 1]
# 返回行的子集，例如 Admit = 0，即*负*示例集
negative = pdData[pdData['admitted'] == 0]

plt.subplots(figsize=(10, 5))
plt.scatter(positive['exam_1'], positive['exam_2'], s=30, c='b', marker='o', label='Admitted')
plt.scatter(negative['exam_1'], negative['exam_2'], s=30, c='r', marker='x', label='Not Admitted')
plt.legend()
plt.xlabel('Exam_1 Score')
plt.ylabel('Exam_2 Score')
plt.show()

# 逻辑回归类
class LogisticRegression:
    """逻辑回归类"""

    def __init__(self, n):
        self.STOP_ITER = 0
        self.STOP_COST = 1
        self.STOP_GRAD = 2
        self.n = n

    def sigmoid(self, z):
        """
            sigmoid函数
            将预测值映射成概率
        """
        return 1 / (1 + np.exp(-z))

    def model(self, X, theta):
        """
            预测函数：返回预测值
        """
        return self.sigmoid(np.dot(X, theta.T))

    def cost(self, X, y, theta):
        """损失函数"""
        left = np.multiply(-y, np.log(self.model(X, theta)))
        right = np.multiply(1 - y, np.log(1 - self.model(X, theta)))
        return np.sum(left - right) / (len(X))

    def gradient(self, X, y, theta):
        """计算梯度"""
        grad = np.zeros(theta.shape)
        error = (self.model(X, theta) - y).ravel()
        for j in range(len(theta.ravel())):  # for each parameter
            term = np.multiply(error, X[:, j])
            grad[0, j] = np.sum(term) / len(X)
        return grad

    def stop_criterion(self, type, value, threshold):
        """
            停止标准函数：
                1.迭代次数
                2.损失值变化
                3.梯度变化
        """
        if type == self.STOP_ITER:
            return value > threshold
        elif type == self.STOP_COST:
            return abs(value[-1] - value[-2]) < threshold
        elif type == self.STOP_GRAD:
            return np.linalg.norm(value) < threshold

    def shuffle_data(self, data):
        """洗牌"""
        np.random.shuffle(data)
        cols = data.shape[1]
        X = data[:, 0:cols - 1]
        y = data[:, cols - 1:]
        return X, y

    def descent(self, data, theta, batchSize, stopType, thresh, alpha):
        """梯度下降求解"""

        init_time = time.time()
        i = 0  # 迭代次数
        k = 0  # batch
        X, y = self.shuffle_data(data)
        grad = np.zeros(theta.shape)  # 计算的梯度
        costs = [self.cost(X, y, theta)]  # 损失值

        while True:
            grad = self.gradient(X[k:k + batchSize], y[k:k + batchSize], theta)
            k += batchSize  # 取batch数量个数据
            if k >= self.n:
                k = 0
                X, y = self.shuffle_data(data)  # 重新洗牌
            theta = theta - alpha * grad  # 参数更新
            costs.append(self.cost(X, y, theta))  # 计算新的损失
            i += 1

            if stopType == self.STOP_ITER:
                value = i
            elif stopType == self.STOP_COST:
                value = costs
            elif stopType == self.STOP_GRAD:
                value = grad
            if self.stop_criterion(stopType, value, thresh):
                break

        return theta, i - 1, costs, grad, time.time() - init_time

    def predict(self, X, theta):
        return [1 if x >= 0.5 else 0 for x in self.model(X, theta)]


# 处理数据
# 在 try / except 结构中，以免在块 si 执行多次时返回错误
pdData.insert(0, 'Ones', 1)

# 设置 X（训练数据）和 y（目标变量）
# 将数据的 Pandas 表示转换为对进一步计算有用的数组
orig_data = pdData.values
cols = orig_data.shape[1]
X = orig_data[:, 0:cols - 1]
y = orig_data[:, cols - 1:cols]

# 转换为 numpy 数组并初始化参数数组 theta
theta = np.zeros([1, 3])

print(X[:5])
print(y[:5])
print(theta)

# 功能函数
lr = LogisticRegression(100)


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = lr.descent(data, theta, batchSize, stopType, thresh, alpha)
    print(theta)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += f" data / learning rate: {alpha} / "
    if batchSize == 1:
        strDescType = "Stochastic"
    elif batchSize == n:
        strDescType = "Gradient"
    else:
        strDescType = f"Mini-batch ({batchSize})"
    name += strDescType + " descent / Stop: "
    if stopType == lr.STOP_ITER:
        strStop = f"{thresh} iterations"
    elif stopType == lr.STOP_COST:
        strStop = f"costs change < {thresh}"
    else:
        strStop = f"gradient norm < {thresh}"
    name += strStop
    #     print(name)
    print(f"{name}\nTheta: {theta} / Iter: {iter} / Last cost: {costs[-1]:03.2f} / Duration: {dur:03.2f}s")
    plt.subplots(figsize=(12, 4))
    plt.plot(np.arange(len(costs)), costs, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.xlim(-1, )
    plt.title(name.upper())
    return theta


print("基于次数的迭代策略的batch梯度下降(5000次)")
n = 100
theta = runExpe(orig_data, theta, n, 0, thresh=2000, alpha=0.000001)
plt.show()

print("基于损失值的迭代策略的batch梯度下降（109901次）")
runExpe(orig_data, theta, n, 1, thresh=0.000001, alpha=0.001)
plt.show()

print("根据梯度变化停止的batch梯度下降（40045次）")
runExpe(orig_data, theta, n, 2, thresh=0.05, alpha=0.001)
plt.show()

print("看一看准确率")
scaled_X = orig_data[:, :3]
y = orig_data[:, 3]
plt.show()

predictions = lr.predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
plt.show()

print("尝试下对数据进行标准化 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1")
scaled_data = orig_data.copy()
scaled_data[:, 1:3] = pp.scale(orig_data[:, 1:3])
plt.show()

print("再基于梯度变化停止的batch梯度下降（139711次）")
runExpe(scaled_data, theta, n, 2, thresh=0.002 * 2, alpha=0.001)
plt.show()

print("基于梯度变化停止的随机梯度下降（72605次）")
theta = runExpe(scaled_data, theta, 1, 2, thresh=0.002 / 5, alpha=0.001)
plt.show()

# print("在看一下准确度")
# scaled_X = scaled_data[:, :3]
# y = scaled_data[:, 3]
#
# predictions = sklearn.svm._libsvm.predict(scaled_X, theta)
# correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
# accuracy = (sum(map(int, correct)) % len(correct))
# print('accuracy = {0}%'.format(accuracy))


print("再基于梯度变化停止的mini-batch的梯度下降（3051次）")
theta = runExpe(scaled_data, theta, 16, 2, thresh=0.002 * 2, alpha=0.001)
plt.show()
