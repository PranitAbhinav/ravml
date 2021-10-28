import matplotlib.pyplot as plt
import ravop.core as R
from ravop.core import Graph
from ravcom import inform_server
inform_server()

class LinearRegression(Graph):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.X=None
        self.y=None
        self.iterations=None
        self.m=None
        self.theta=None

        self.op_theta=None


    def fit(self,X,y,theta,iterations,alpha=0.001,sample_weight=None):
        self.raw_X = X
        self.raw_y = y
        self.X=X
        self.y=y
        if isinstance(self.m,R.Scalar) is False:
            self.m = R.Scalar(self.raw_y.shape[0])
        else:
            self.m=R.Scalar(self.raw_y.shape[0])
        self.alpha=alpha
        self.iterations=iterations
        if isinstance(self.raw_X,R.Tensor) is False:
            self.X = R.Tensor(self.raw_X.tolist())
        if isinstance(self.raw_y,R.Tensor) is False:
            self.y = R.Tensor(self.raw_y.tolist())
        if isinstance(self.theta,R.Tensor) is False:
            self.theta = R.Tensor(theta.tolist())
        self.compute_cost()
        self.theta, self.op_theta[0], self.op_theta[1] =self.gradient_descent(self.alpha,self.iterations)
        return self

    def compute_cost(self):
        residual = self.X.dot(self.theta).sub(self.y)
        return (R.Scalar(1).div(R.Scalar(2).multiply(self.m))).multiply(residual.dot(residual.transpose()))

    def gradient_descent(self, alpha, num_iters):
        alpha_ = R.Scalar(alpha)
        for e in range(num_iters):
            residual = self.X.dot(self.theta).sub(self.y)
            while residual.status != 'computed':
                pass
            temp = self.theta.sub((alpha_.div(self.m)).multiply(self.X.transpose().dot(residual)))
            while temp.status!='computed':
                #print(temp.status)
                pass
            self.theta = temp
            print('Iteration : ',e)
        self.op_theta = self.theta()
        print('Theta found by gradient descent: intercept={0}, slope={1}'.format(self.op_theta[0],self.op_theta[1]))
        return self.theta, self.op_theta[0], self.op_theta[1]

    def plot_graph(self,optimal_theta, res_file_path):
        optimal_theta = optimal_theta()
        fig, ax = plt.subplots()
        ax.plot(self.raw_X[:,1], self.raw_y[:,0], 'o', label='Raw Data')
        ax.plot(self.raw_X[:,1], self.raw_X.dot(optimal_theta), linestyle='-', label='Linear Regression')
        plt.ylabel('Profit')
        plt.xlabel('Population of City')
        legend = ax.legend(loc='upper center', shadow=True)
        plt.savefig(res_file_path)
        plt.show()


    def predict(self,x_test):
        if isinstance(x_test,R.Tensor) is False:
            x_test=R.Tensor(x_test)
        res= x_test.dot(R.Tensor(self.op_theta[1])).add(R.Tensor(self.op_theta[0]))
        res.wait()
        return res
        pass

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

