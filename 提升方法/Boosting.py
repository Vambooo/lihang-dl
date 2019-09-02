import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate


# 加载数据
dataset = pd.read_csv('../data/mushrooms.csv', header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['target', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                   'gill-spacing',
                   'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                   'stalk-surface-below-ring', 'stalk-color-above-ring',
                   'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                   'population',
                   'habitat']
# 由于sklearn DecisionTreeClassifier仅采用数值，因此将字符串中的要素值编码为整数
for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

Tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
X = dataset.drop('target', axis=1)
Y = dataset['target'].where(dataset['target'] == 1, -1)
predictions = np.mean(cross_validate(Tree_model, X, Y, cv=100)['test_score'])
print('The accuracy is: ', predictions * 100, '%')


class Boosting:
    def __init__(self, dataset, T, test_dataset):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None

    def fit(self):
        # Set the descriptive features and the target feature
        X = self.dataset.drop(['target'], axis=1)
        Y = self.dataset['target'].where(self.dataset['target'] == 1, -1)
        # 初始化每个样本的权重 wi = 1/N ，并创建一个计算评估的DataFrame
        Evaluation = pd.DataFrame(Y.copy())
        Evaluation['weights'] = 1 / len(self.dataset)  # 初始化权值为 w = 1/N

        alphas = []
        models = []

        for t in range(self.T):
            # 训练决策树 Stump(s)
            Tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)  #

            """
            在加权数据集上训练决策树分类器，其中权重是取决于先前决策树训练的结果，
            为此，这里使用上述创建的评估DataFrame的权重和fit方法的sample_weights参数，
            该参数序列如果为None,则表示样本的权重相等，如果不是None，则表示样本的权重不均等。
            """
            model = Tree_model.fit(X, Y, sample_weight=np.array(Evaluation['weights']))

            # 将单个弱分类器附加到列表中，该列表稍后用于进行加权决策
            models.append(model)
            predictions = model.predict(X)
            score = model.score(X, Y)
            # 将值添加到评估 DataFrame中
            Evaluation['predictions'] = predictions
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation['target'], 1, 0)
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation['target'], 1, 0)
            # 计算误分类率和准确性
            accuracy = sum(Evaluation['evaluation']) / len(Evaluation['evaluation'])
            misclassification = sum(Evaluation['misclassified']) / len(Evaluation['misclassified'])
            # 计算错误率
            err = np.sum(Evaluation['weights'] * Evaluation['misclassified']) / np.sum(Evaluation['weights'])

            # 计算alpha值
            alpha = np.log((1 - err) / err)
            alphas.append(alpha)
            # 更新权重 wi --> 这些更新后的权重值在sample_weight参数中用于训练写一个决策树分类器
            Evaluation['weights'] *= np.exp(alpha * Evaluation['misclassified'])

        self.alphas = alphas
        self.models = models

    def predict(self):
        X_test = self.test_dataset.drop(['target'], axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset['target'].reindex(range(len(self.test_dataset))).where(self.dataset['target'] == 1,
                                                                                          -1)

        # 对于self.model列表中的每个模型，进行预测
        accuracy = []
        predictions = []

        for alpha, model in zip(self.alphas, self.models):
            prediction = alpha * model.predict(X_test)  # 对列表中的单个决策树分类器模型使用预测方法
            predictions.append(prediction)
            self.accuracy.append(
                np.sum(np.sign(np.sum(np.array(predictions), axis=0)) == Y_test.values) / len(predictions[0]))

        self.predictions = np.sign(np.sum(np.array(predictions), axis=0))


# 根据使用的模型数量绘制模型精度
number_of_base_learners = 50
fig = plt.figure(figsize=(10, 10))
ax0 = fig.add_subplot(111)
for i in range(number_of_base_learners):
    model = Boosting(dataset, i, dataset)
    model.fit()
    model.predict()
ax0.plot(range(len(model.accuracy)), model.accuracy, '-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('数量为 ', number_of_base_learners, 'base models ，获得的精度为 ', model.accuracy[-1] * 100,
      '%')

plt.show()