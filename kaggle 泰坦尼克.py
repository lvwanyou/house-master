# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:33:16 2017

@author: user
"""

import pandas as pd
import numpy as np

# 导入原始数据
data = pd.read_csv('C:/Users/lenovo/Desktop/house-master/titanic_train.csv')

# 描述性统计
data.head()
print (data.describe())

# Age字段缺失值填充为均值
data['Age'] = data['Age'].fillna(data['Age'].mean())
print (data.describe())

# Sex字段转为数值男0女1
data['Sex'] = data['Sex'].replace(('male', 'female'), ('0', '1'))

# Embarked字段处理
data['Embarked'] = data['Embarked'].fillna('S')
data['Embarked'] = data['Embarked'].replace(('S', 'C', 'Q'), ('0', '1', '2'))

# 线性回归
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

alg = LinearRegression()
kf = KFold(data.shape[0], n_folds=3, random_state=1)

# 划定训练集
predictions = []
for train, test in kf:
    # 取出训练数据，在指定特征中
    train_predictors = (predictors.iloc[train, :])
    train_target = (data['Survived'].iloc[train])
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(predictors.iloc[test, :])
    predictions.append(test_predictions)

# 线性回归跑分
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
mingzhonglv = sum(predictions[predictions == data['Survived']]) / len(predictions)

# 逻辑回归
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

# CV为交叉验证划分数量，默认为3
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(
    alg, predictors, data['Survived'], cv=3)
print (scores.mean())

# 随机森林
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# n_estimators表示多少颗树，min_samples_split=2,min_samples_leaf=1分裂停止条件
# （2个样本停止，最小叶子节点1个停止）
alg = RandomForestClassifier(random_state=1, n_estimators=10
                             , min_samples_split=2, min_samples_leaf=1)

kf = cross_validation.KFold(data.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, data[predictors], data['Survived'], cv=kf)
print (scores.mean())

# 随机森林继续优化，增加跑分
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# 随机森林调参，跑分增加到0.83
alg = RandomForestClassifier(random_state=1, n_estimators=50
                             , min_samples_split=4, min_samples_leaf=2)

kf = cross_validation.KFold(data.shape[0], n_folds=3, random_state=1)

scores = cross_validation.cross_val_score(alg, data[predictors], data['Survived'], cv=kf)
print (scores.mean())

# 特征优化
# 家庭成员数目
data['Familysize'] = data['SibSp'] + data['Parch']
# 名字长度
data['NameLength'] = data['Name'].apply(lambda x: len(x))

# 选出重要特征
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

selector = SelectKBest(f_classif, k=5)
selector.fit(data[predictors], data['Survived'])

scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
