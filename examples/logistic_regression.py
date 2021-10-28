from ravml.linear.logistic_regression import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]
y = (iris.target != 0) * 1
print(X,y)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


X, y = load_iris(return_X_y=True)

a=LogisticRegression(num_iter=50,fit_intercept=False)

X,y=np.array(X[:100]),np.array(y[:100])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

print(X_train,y_train)

a.fit(np.array(X_train),np.array(y_train))

y_pred=a.predict(X_test)

print((y_pred))

acc=a.accuracy(y_test,y_pred)

print(acc)

'''model = LogisticRegression(lr=0.1, num_iter=30)

model.fit(X, y)

preds = model.predict(X)
print((preds == y).mean())
print(model.theta())

model.plot_loss()

model.visualize(X,y)'''
