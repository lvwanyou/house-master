# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:07:17 2017

@author: user
"""

from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print (clf.predict([[140, 1]]))

import numpy as np

a = np.array([[0, 1, 2, 3, 4],
              [9, 8, 7, 6, 5]])
a.shape

import numpy as np

a = np.array([[0, 1, 2, 3, 4],
              [9, 8, 7, 6, 5]])
a.itemsize

b = np.random.shuffle(a)
print (b)

a = np.add_docstring((3, 4), dtype='int32')
