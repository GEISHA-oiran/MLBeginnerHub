# 导入数据和包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from math import log

# 生成数据
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1208)


# 下面给出完整的，封装的决策树代码
# 可以像调用sklearn的决策树一样调用它。
class Node:
    def __init__(self, x_data, y_label, dimension, value):
        self.x_data = x_data
        self.y_label = y_label
        self.dimension = dimension  # 特征的维度索引
        self.value = value  # 划分特征的数值
        self.left = None
        self.right = None


class DTree:
    def __init__(self):
        self.root = None

    def fit(self, x_train, y_train):
        # 计算熵
        def entropy(y_label):
            counter = Counter(y_label)  # 使用Counter类统计y_label中每个类别出现的次数，返回一个类似字典的对象，其中键是类别，值是出现次数
            ent = 0.0  # 初始化熵值为0
            for num in counter.values():  # 遍历每个类别出现的次数
                p = num / len(y_label)  # 计算每个类别在标签集合中的比例
                ent += -p * log(p)  # 根据熵的计算公式，累加每个类别对总熵的贡献
            return ent  # 返回计算得到的熵值

        # 划分一次数据集
        # 遍历所有维度的特征，不断寻找一个合适地划分数值，找到能把熵降到最低的那个特征和数值
        def one_split(x_data, y_label):
            best_entropy = float('inf')  # 初始化最佳熵为正无穷
            best_dimension = -1  # 初始化最佳特征维度索引为-1
            best_value = -1  # 初始化最佳切分值为-1

            for d in range(x_data.shape[1]):  # 遍历特征的维度
                sorted_index = np.argsort(x_data[:, d])  # 对特征值进行排序并获取排序后的索引 在决策树算法中，我们需要对特征的取值进行排序，以便找到最佳的切分点。
                for i in range(1, len(x_data)):  # 遍历特征值
                    if x_data[sorted_index[i], d] != x_data[sorted_index[i - 1], d]:  # 如果特征值不同
                        value = (x_data[sorted_index[i], d] + x_data[sorted_index[i - 1], d]) / 2  # 计算切分值为中间值
                        x_left, x_right, y_left, y_right = split(x_data, y_label, d, value)  # 使用 split 函数进行数据集划分

                        p_left = len(x_left) / len(x_data)  # 计算左子树占比
                        p_right = len(x_right) / len(x_data)  # 计算右子树占比

                        ent = p_left * entropy(y_left) + p_right * entropy(y_right)  # 计算划分后的熵
                        if ent < best_entropy:  # 如果当前熵值优于最佳熵
                            best_entropy = ent  # 更新最佳熵
                            best_dimension = d  # 更新最佳特征维度
                            best_value = value  # 更新最佳切分值
            return best_entropy, best_dimension, best_value

        # 划分数据集
        def split(x_data, y_label, dimension, value):
            """
            x_data:输入特征
            y_label:输入标签类别
            dimension:选取输入特征的维度索引
            value：划分特征的数值

            return 左子树特征，右子树特征，左子树标签，右子树标签
            """
            index_left = (x_data[:, dimension] <= value)
            index_right = (x_data[:, dimension] > value)
            return x_data[index_left], x_data[index_right], y_label[index_left], y_label[index_right]

        def create_tree(x_data, y_label):
            ent, dim, value = one_split(x_data, y_label)  # 找到最佳的划分特征和划分值
            x_left, x_right, y_left, y_right = split(x_data, y_label, dim, value)  # 根据最佳特征和划分值进行数据集划分
            node = Node(x_data, y_label, dim, value)  # 创建一个节点，记录特征、划分值和数据集信息
            if ent < 0.000000001:  # 如果当前节点的熵值小于阈值
                return node  # 则将当前节点作为叶子节点返回
            node.left = create_tree(x_left, y_left)  # 递归构建左子树
            node.right = create_tree(x_right, y_right)  # 递归构建右子树
            return node  # 返回当前节点

        self.root = create_tree(x_train, y_train)  # 入口

        return self

    def predict(self, x_predict):
        def travel(x_data, node):
            p = node
            if x_data[p.dimension] <= p.value and p.left:  # 如果当前特征值小于等于划分值且存在左子树
                pred = travel(x_data, p.left)  # 递归进入左子树
            elif x_data[p.dimension] > p.value and p.right:  # 如果当前特征值大于划分值且存在右子树
                pred = travel(x_data, p.right)  # 递归进入右子树
            else:  # 如果当前节点为叶子节点
                counter = Counter(p.y_label)  # 统计叶子节点中类别标签出现的次数
                pred = counter.most_common(1)[0][0]  # 返回出现次数最多的类别标签作为预测结果
            return pred

        y_predict = []
        for data in x_predict:
            y_pred = travel(data, self.root)  # 对每个样本进行预测
            y_predict.append(y_pred)  # 将预测结果添加到列表中
        return np.array(y_predict)  # 返回预测结果数组

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return np.sum(y_predict == y_test) / len(y_predict)

    def __repr__(self):
        return "DTree(criterion='entropy')"


dt = DTree()
dt.fit(x_train, y_train)
print(dt.score(x_test, y_test))
