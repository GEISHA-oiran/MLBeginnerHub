import numpy as np
import matplotlib.pyplot as plt

#  1.导入数据（data.csv)-----------------------------------------------------
points = np.genfromtxt('线性回归.csv', delimiter=',')

# points
# 提取 points 中的两对数据，分别作为 x, y
# points[0][0]  等价于
# points[0,0]  # 第一行第一列数据
# points[0,0:1] # array([32.50234527])
# points[0,0:2] # 第一行数据 array([32.50234527, 31.70700585])
# points[0,0:] # 第一行数据 array([32.50234527, 31.70700585])
x = points[:, 0]  # 第一列数据
y = points[:, 1]  # 第二列数据

# 用 scatter 画出散点图
plt.scatter(x, y)
plt.show()


#  2.定义损失函数------------------------------------------------------------
# 损失函数是模型系数的函数，还需要传入数据的 x,y
def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)
    # 逐点计算【实际数据 yi 与 模型数据 f(xi) 的差值】的平方，然后求平均
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2

    return total_cost / M


#  3.定义模型的超参数------------------------------------------------------------
alpha = 0.0001
initial_w = 0
initial_b = 0
num_iter = 10


#  4.定义核心梯度下降模型函数------------------------------------------------------------
def grad_desc(points, initial_w, initial_b, alpha, num_iter):
    w = initial_w
    b = initial_b
    # 定义一个list保存所有的损失函数值，用来显示下降的过程
    cost_list = []

    for i in range(num_iter):
        # 先计算初始值的损失函数的值
        cost_list.append(compute_cost(w, b, points))
        w, b = step_grad_desc(w, b, alpha, points)

    return [w, b, cost_list]


def step_grad_desc(current_w, current_b, alpha, points):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(points)

    # 对每一个点带入公式求和
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += (current_w * x + current_b - y)

    # 用公式求当前梯度
    grad_w = 2 / M * sum_grad_w
    grad_b = 2 / M * sum_grad_b

    # 梯度下降，更新当前的 w 和 b
    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b

    return updated_w, updated_b


#  5.测试：运行梯度下降算法，计算最优的 w 和 b------------------------------------------------------------
w, b, cost_list = grad_desc(points, initial_w, initial_b, alpha, num_iter)

print('w is:', w)
print('b is:', b)

cost = compute_cost(w, b, points)
print('cost is:', cost)

plt.plot(cost_list)
plt.show()

#  6.画出拟合曲线------------------------------------------------------------
# 先用 scatter 画出2维散点图
plt.scatter(x, y)

# 针对每一个x，计算出预测的值
pred_y = w * x + b
# 再用 plot 画出2维直线图
plt.plot(x, pred_y, c='r')
plt.show()
