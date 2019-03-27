import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

def bootstrap(model,D,target_name):
    rows,__ = D.shape
    acc_list = []
    for i in range(200):
        B = D.sample(n=rows,replace=True)
        X = B.drop(target_name,1)
        y = B[target_name]
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)
        model.fit(train_X, train_y)
        predict_y = model.predict(test_X)
        acc_list.append(accuracy_score(test_y, predict_y))
    acc_list.sort()
    ub = np.percentile(acc_list,97.5)
    lb = np.percentile(acc_list,2.5)
    return (lb, ub)
