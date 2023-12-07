# -*- coding: utf-8 -*-
import csv
from random import seed
from random import randrange


# 加载数据
# 按照常规的文件读取和处理流程来组织代码：首先打开文件，然后将其内容解析为适当的数据结构，最后对数据进行处理或返回。
def loadCSV(filename):  # 定义loadCSV函数，参数为filename
    dataSet = []  # 创建空列表dataSet
    with open(filename, 'r') as file:  # 用只读方式打开filename文件并赋值给file
        csvReader = csv.reader(file)  # 用csv模块的reader方法读取file文件并赋值给csvReader
        for line in csvReader:  # 遍历csvReader中的每一行
            dataSet.append(line)  # 将每一行添加到dataSet列表中
    return dataSet  # 返回dataSet列表


# 除了判别列，其他列都转换为float类型
def column_to_float(dataSet):  # 定义column_to_float函数，参数为dataSet
    featLen = len(dataSet[0]) - 1  # 获取dataSet中最后一列的索引
    for data in dataSet:  # 遍历dataSet中的每一行
        for column in range(featLen):  # 遍历每一行中除最后一列之外的每一列
            data[column] = float(data[column].strip())  # 将每一列的字符串转换为浮点数并赋值回原位置


# 将数据集分成N块，方便交叉验证
def spiltDataSet(dataSet, n_folds):
    fold_size = int(len(dataSet) / n_folds)  # 计算每份数据的大小
    dataSet_copy = list(dataSet)  # 复制数据集
    dataSet_spilt = []  # 创建用于存储分割后数据集的列表
    for i in range(n_folds):  # 循环n_folds次，将数据集分割成n份
        fold = []  # 创建一个空列表用于存储每份数据
        while len(fold) < fold_size:  # 当该份数据未达到应有的大小时
            index = randrange(len(dataSet_copy) - 1)  # 随机选择数据集中的索引（有放回）
            fold.append(dataSet_copy.pop(index))  # 将随机选择的数据加入该份数据，并从数据集中移除
        dataSet_spilt.append(fold)  # 将该份数据添加到分割后数据集列表中
    return dataSet_spilt  # 返回分割后的数据集列表


# 构造数据子集
def get_subsample(dataSet, ratio):
    subdataSet = []  # 创建空列表用于存储子样本
    lenSubdata = round(len(dataSet) * ratio)  # 计算子样本的大小
    while len(subdataSet) < lenSubdata:  # 当子样本未达到应有大小时
        index = randrange(len(dataSet) - 1)  # 随机选择数据集中的索引（有放回）
        subdataSet.append(dataSet[index])  # 将随机选择的数据加入子样本
    return subdataSet  # 返回子样本


# 分割数据集
def data_spilt(dataSet, index, value):
    left = []  # 创建存储左子集的列表
    right = []  # 创建存储右子集的列表
    for row in dataSet:  # 遍历数据集中的每一行
        if row[index] < value:  # 如果行中指定索引的值小于给定值
            left.append(row)  # 将该行添加到左子集
        else:
            right.append(row)  # 否则将该行添加到右子集
    return left, right  # 返回左右的子集


# 计算分割代价
def spilt_loss(left, right, class_values):
    loss = 0.0  # 初始化分割代价为0
    for class_value in class_values:  # 遍历每个类别的取值
        left_size = len(left)  # 计算左子集的大小
        if left_size != 0:  # 防止除数为零
            prop = [row[-1] for row in left].count(class_value) / float(left_size)  # 计算左子集中该类别的占比
            loss += (prop * (1.0 - prop))  # 根据占比计算损失
        right_size = len(right)  # 计算右子集的大小
        if right_size != 0:  # 防止除数为零
            prop = [row[-1] for row in right].count(class_value) / float(right_size)  # 计算右子集中该类别的占比
            loss += (prop * (1.0 - prop))  # 根据占比计算损失
    return loss  # 返回分割代价


# 选取任意的n个特征，在这n个特征中，选取分割时的最优特征
def get_best_spilt(dataSet, n_features):
    features = []  # 用于存储随机选择的特征的列表
    class_values = list(set(row[-1] for row in dataSet))  # 获取数据集中的类别值
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None  # 初始化最佳分割的特征索引、值、损失和左右子集
    while len(features) < n_features:  # 当特征列表中的特征数量小于n_features时
        index = randrange(len(dataSet[0]) - 1)  # 随机选择一个特征索引
        if index not in features:  # 如果选择的特征索引不在特征列表中
            features.append(index)  # 将特征索引添加到特征列表中
    for index in features:  # 遍历随机选择的特征
        for row in dataSet:  # 遍历数据集中的每一行
            left, right = data_spilt(dataSet, index, row[index])  # 根据特征和特征值将数据集分割成左右子集
            loss = spilt_loss(left, right, class_values)  # 计算分割代价
            if loss < b_loss:  # 如果当前分割代价比最小代价还小
                b_index, b_value, b_loss, b_left, b_right = index, row[index], loss, left, right  # 更新最佳分割信息
    return {'index': b_index, 'value': b_value, 'left': b_left, 'right': b_right}  # 返回最佳分割特征的索引、值、左右子集


# 决定输出标签
def decide_label(data):
    output = [row[-1] for row in data]  # 获取数据中所有标签的列表
    return max(set(output), key=output.count)  # 返回出现次数最多的标签


# 子分割，不断地构建叶节点的过程
def sub_spilt(root, n_features, max_depth, min_size, depth):
    left = root['left']
    right = root['right']
    del (root['left'])
    del (root['right'])
    if not left or not right:
        root['left'] = root['right'] = decide_label(left + right)  # 如果左节点或右节点为空，则将左右节点设为出现次数最多的标签
        return
    if depth > max_depth:
        root['left'] = decide_label(left)  # 如果深度超过最大深度，则将左节点设为出现次数最多的标签
        root['right'] = decide_label(right)  # 将右节点设为出现次数最多的标签
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)  # 如果左节点数据量小于最小数据量，则将左节点设为出现次数最多的标签
    else:
        root['left'] = get_best_spilt(left, n_features)  # 否则，根据最佳分割点构建左节点
        sub_spilt(root['left'], n_features, max_depth, min_size, depth + 1)  # 递归构造左子树
    if len(right) < min_size:
        root['right'] = decide_label(right)  # 如果右节点数据量小于最小数据量，则将右节点设为出现次数最多的标签
    else:
        root['right'] = get_best_spilt(right, n_features)  # 否则，根据最佳分割点构建右节点
        sub_spilt(root['right'], n_features, max_depth, min_size, depth + 1)  # 递归构造右子树

    # 构造决策树


def build_tree(dataSet, n_features, max_depth, min_size):
    root = get_best_spilt(dataSet, n_features)  # 找到最佳分割点作为根节点
    sub_spilt(root, n_features, max_depth, min_size, 1)  # 构建整棵决策树
    return root  # 返回根节点


def predict(tree, row):
    predictions = []  # 初始化一个存储预测结果的列表
    if row[tree['index']] < tree['value']:  # 如果数据的某个特征值小于当前节点的划分值
        if isinstance(tree['left'], dict):  # 如果左子树仍然是一个字典（即非叶子节点）
            return predict(tree['left'], row)  # 递归地对左子树进行预测
        else:  # 如果左子树是叶子节点
            return tree['left']  # 返回左子树的预测结果
    else:  # 如果数据的某个特征值大于或等于当前节点的划分值
        if isinstance(tree['right'], dict):  # 如果右子树仍然是一个字典（即非叶子节点）
            return predict(tree['right'], row)  # 递归地对右子树进行预测
        else:  # 如果右子树是叶子节点
            return tree['right']  # 返回右子树的预测结果


# predictions=set(predictions)


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]  # 使用每棵树对数据进行预测，并将预测结果存储在列表中
    # 找到预测结果中出现次数最多的值作为最终预测结果
    return max(set(predictions), key=predictions.count)  # 找到预测结果中出现次数最多的值作为最终预测结果


# 创建随机森林
def random_forest(train, test, ratio, n_feature, max_depth, min_size, n_trees):
    trees = []  # 用于存储每棵树的列表
    for i in range(n_trees):
        train = get_subsample(train, ratio)  # 从训练集中进行子采样
        tree = build_tree(train, n_features, max_depth, min_size)  # 构建一棵树
        print('tree %d: '%i,tree)
        trees.append(tree)  # 将构建好的树添加到列表中
    # predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]  # 对测试集中的每一行数据进行预测
    return predict_values  # 返回预测结果列表


# 计算准确率
def accuracy(predict_values, actual):
    correct = 0  # 初始化正确预测的数量
    for i in range(len(actual)):  # 遍历每一个预测结果
        if actual[i] == predict_values[i]:  # 如果预测结果与实际结果相符
            correct += 1  # 正确预测数量加一
    return correct / float(len(actual))  # 返回正确预测的比例


if __name__ == '__main__':
    seed(1)  # 设置随机种子，以确保结果的可重复性
    dataSet = loadCSV('sonar-all-data.csv')  # 载入数据集
    column_to_float(dataSet)  # 将数据集中的数据类型转换为浮点型
    n_folds = 5  # 设置交叉验证的折数
    max_depth = 15  # 决策树的最大深度
    min_size = 1  # 叶子节点的最小样本数
    ratio = 1  # 子采样比例
    n_features = 15  # 特征数量
    n_trees = 20 # 随机森林中树的数量
    folds = spiltDataSet(dataSet, n_folds)  # 将数据集划分为指定折数的子集
    scores = []  # 用于存储每次交叉验证的准确率
    for fold in folds:  # 对每个子集进行交叉验证
        train_set = folds[:]  # 复制数据集
        train_set.remove(fold)  # 从训练集中移除当前折的数据
        train_set = sum(train_set, [])  # 将多个fold列表组合成一个train_set列表
        test_set = [list(row)[:-1] for row in fold]  # 从当前折中提取测试数据
        actual = [row[-1] for row in fold]  # 获取当前折的实际结果
        predict_values = random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)  # 使用随机森林进行预测
        accur = accuracy(predict_values, actual)  # 计算准确率
        scores.append(accur)  # 将准确率添加到列表中
    print('Trees is %d' % n_trees)  # 打印树的数量
    print('scores:%s' % scores)  # 打印每次交叉验证的准确率
    print('mean score:%s' % (sum(scores) / float(len(scores))))  # 打印平均准确率
