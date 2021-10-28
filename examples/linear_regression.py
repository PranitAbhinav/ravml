from ravml.linear import LinearRegression
import numpy as np
import pathlib
#from sklearn.linear_model import LinearRegression

def preprocess(data):
    x = data[:,0]
    y = data[:,1]
    y = y.reshape(y.shape[0], 1)
    x = np.c_[np.ones(x.shape[0]), x] # adding column of ones to X to account for theta_0 (the intercept)
    theta = np.zeros((2, 1))
    return x,y,theta

iterations = 20
alpha = 0.01

data = np.loadtxt('data_linreg.txt', delimiter=',')

x,y,theta = preprocess(data)

model = LinearRegression()
model.fit(x,y,theta, iterations)            # initial cost with coefficients at zero
print(model.theta, model.op_theta[0], model.op_theta[1])
#ypred=model.predict(x)
#print(ypred)
model.plot_graph(model.theta, 'result.png')

