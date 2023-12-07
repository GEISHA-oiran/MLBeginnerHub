from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

california = fetch_california_housing()
# 划分数据集
X, y = california.data, california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)  # random_state是设置随机种子数
st = StandardScaler()
st.fit(X_train)
X_train = st.transform(X_train)
X_test = st.transform(X_test)
for kernel in ['linear', 'rbf']:
    svr = SVR(kernel=kernel)
    svr.fit(X_train, y_train)
    print(kernel, "核函数训练集：", svr.score(X_train, y_train))
    print(kernel, "核函数测试集：", svr.score(X_test, y_test))
