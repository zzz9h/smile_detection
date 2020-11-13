# pandas 读取 CSV
import pandas as pd

# 分割数据
from sklearn.model_selection import train_test_split

# 用于数据预加工标准化
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier        # 神经网络模型中的多层网络模型
from sklearn.externals import joblib

# 从 csv 读取数据
def pre_data():
    # 41 维表头
    column_names = []
    for i in range(0, 40):
        column_names.append("feature_" + str(i + 1))
    column_names.append("output")

    # read csv
    rd_csv = pd.read_csv("data/data_csvs/data.csv", names=column_names)

    # 输出 csv 文件的维度
    # print("shape:", rd_csv.shape)

    X_train, X_test, y_train, y_test = train_test_split(

        # input 0-40
        # output 41
        rd_csv[column_names[0:40]],
        rd_csv[column_names[40]],

        # 25% for testing, 75% for training
        test_size=0.25,
        random_state=33)

    return X_train, X_test, y_train, y_test


path_models = "data/data_models/"

# MLPC, Multi-layer Perceptron Classifier, 多层感知机分类（神经网络）
def model_MLPC():
    # get data
    X_train_MLPC, X_test_MLPC, y_train_MLPC, y_test_MLPC = pre_data()

    # 数据预加工
    ss_MLPC = StandardScaler()
    X_train_MLPC = ss_MLPC.fit_transform(X_train_MLPC)
    X_test_MLPC = ss_MLPC.transform(X_test_MLPC)

    # 初始化 MLPC
    MLPC = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)

    # 调用 MLPC 中的 fit() 来训练模型参数
    MLPC.fit(X_train_MLPC, y_train_MLPC)

    # save MLPC model
    joblib.dump(MLPC, path_models + "model_MLPC1.m")

    # 评分函数
    score_MLPC = MLPC.score(X_test_MLPC, y_test_MLPC)
    # print("The accurary of MLPC:", score_MLPC)

    return (ss_MLPC)


model_MLPC()
