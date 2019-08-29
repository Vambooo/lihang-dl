"""
API Reference: http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
定义一个MultinomialNB类
"""
import numpy as np

class MultinomialNB(object):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        """
        多项式模型的朴素贝叶斯分类器。多项式朴素贝叶斯分类器适用于具有离散特征的分类。

        参数
        ----------
        alpha : 平滑参数(float类型)，默认为1.0;
        如果alpha=0则不平滑。
        如果 0 < alpha < 1 则为Lidstone平滑
        如果 alpha = 1 则为Laplace 平滑

        fit_prior : 布尔型
        是否学习类别先验概率。
        如果设置为False，将使用统一的优先权。

        class_prior : array-like, size (n_classes,)
                类别的先验概率。如果指定，则不会根据数据调整优先级。
        """
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        self.conditional_prob = None

    def _calculate_feature_prob(self, feature):
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            value_prob[v] = ((np.sum(np.equal(feature, v)) + self.alpha) / (total_num + len(values) * self.alpha))
        return value_prob

    def fit(self, X, y):
        # TODO: check X,y

        self.classes = np.unique(y)
        # 计算类别先验概率: P(y=ck)
        if self.class_prior == None:
            class_num = len(self.classes)
            if not self.fit_prior:
                self.class_prior = [1.0 / class_num for _ in range(class_num)]  # uniform prior
            else:
                self.class_prior = []
                sample_num = float(len(y))
                for c in self.classes:
                    c_num = np.sum(np.equal(y, c))
                    self.class_prior.append((c_num + self.alpha) / (sample_num + class_num * self.alpha))

        # 计算条件概率 P( xj | y=ck )
        self.conditional_prob = {}  # like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(len(X[0])):  # for each feature
                feature = X[np.equal(y, c)][:, i]
                self.conditional_prob[c][i] = self._calculate_feature_prob(feature)
        return self

    # 给定values_prob {value0:0.2,value1:0.1,value3:0.3,.. } and target_value
    # 返回target_value的概率
    def _get_xj_prob(self, values_prob, target_value):
        return values_prob[target_value]

    # 基于(class_prior,conditional_prob)预测单个样本
    def _predict_single_sample(self, x):
        label = -1
        max_posterior_prob = 0

        # 对于每个类别，计算其后验概率: class_prior * conditional_prob
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
            current_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for feature_i in feature_prob.keys():
                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i], x[j])
                j += 1

            # 比较后验概率并更新最大后验概率，标签
            if current_class_prior * current_conditional_prob > max_posterior_prob:
                max_posterior_prob = current_class_prior * current_conditional_prob
                label = self.classes[c_index]
        return label

    # 样本预测(也可以是单样本预测)
    def predict(self, X):
        # TODO1:check and raise NoFitError
        # ToDO2:check X
        if X.ndim == 1:
            return self._predict_single_sample(X)
        else:
            # 为每个样本进行分类
            labels = []
            for i in range(X.shape[0]):
                label = self._predict_single_sample(X[i])
                labels.append(label)
            return labels


X = np.array([
        [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
        [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
    ])
X = X.T
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

nb = MultinomialNB(alpha=1.0,fit_prior=True)
nb.fit(X,y)
print(nb.predict(np.array([2,4])))#输出-1
