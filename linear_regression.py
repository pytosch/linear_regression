import numpy as np


class LinearRegression():
    
    def __init__(self):
        self.theta_gd = 0
        self.theta_nm = 0
        self.theta_ne = 0
        
        self.train_loss_gd = []
        self.train_loss_nm = []
        self.train_loss_ne = 0
    
    def _adjust_X(self, X):
        # reshape if 1D
        if X.ndim <= 1: X = X.reshape(-1, 1)
        # add x_0 = 1 for all samples
        X = np.hstack([X, np.ones(shape=(X.shape[0], 1))])
        return X
    
    def _adjust_y(self, y):
        # reshape if 1D
        if y.ndim <= 1: y = y.reshape(-1, 1)
        return y
        
    def _predict(self, X, theta):
        return X.dot(theta.T)
    
    
    def _mse(self, y, h):
        # mean squared error
        return 0.5 * ((h-y).T).dot(h-y) / y.shape[0]  
        
    
    def fit_gd(self, X, y, e, lr):
        print('BATCH GRADIENT DESCENT')
        # reshape, include x0
        X = self._adjust_X(X)
        y = self._adjust_y(y)
        # init theta
        self.theta_gd = np.zeros(shape=(y.shape[1], X.shape[1]))
        
        # fit
        for i in range(e):
            # predict
            h = self._predict(X, self.theta_gd)
            # compute loss
            j = self._mse(y, h)
            self.train_loss_gd.append(j)
            # print loss
            print('Epoch {} - Loss: {:.6f}'.format(i, j.squeeze()))
            
            # update
            self.theta_gd -= lr * (h-y).T.dot(X) / y.shape[0]
        
        # final prediction, loss   
        h = self._predict(X, self.theta_gd)
        j = self._mse(y, h)
        self.train_loss_gd.append(j)
        print('Epoch {} - Loss: {:.6f}'.format(i+1, j.squeeze())) 
        
        
    def fit_nm(self, X, y, e):
        print('NEWTONS METHOD')
        # reshape, include x0
        X = self._adjust_X(X)
        y = self._adjust_y(y)
        # init theta
        self.theta_nm = np.zeros(shape=(y.shape[1], X.shape[1]))
        # inverse hessian matrix
        H_inv = np.linalg.inv(X.T.dot(X))
        
        # fit
        for i in range(e):
            # predict
            h = self._predict(X, self.theta_nm)
            # compute loss
            j = self._mse(y, h)
            self.train_loss_nm.append(j)
            # print loss
            print('Epoch {} - Loss: {:.6f}'.format(i, j.squeeze()))
            
            # update
            self.theta_nm -= H_inv.dot(X.T.dot(h-y)).T #/ y.shape[0]
        
        # final prediction, loss   
        h = self._predict(X, self.theta_nm)
        j = self._mse(y, h)
        self.train_loss_nm.append(j)
        print('Epoch {} - Loss: {:.6f}'.format(i+1, j.squeeze()))
        
        
    def fit_ne(self, X, y):
        print('NORMAL EQUATION')
        # reshape, include x0
        X = self._adjust_X(X)
        y = self._adjust_y(y)
        
        # fit
        self.theta_ne = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y).T
        
        # final prediction, loss   
        h = self._predict(X, self.theta_ne)
        j = self._mse(y, h)
        self.train_loss_ne = j
        print('Loss: {:.6f}'.format(j.squeeze()))
    
    
    def test_gd(self, X, y):
        X = self._adjust_X(X)
        y = self._adjust_y(y)
        h = self._predict(X, self.theta_gd)
        j = self._mse(y, h)
        print('Test Loss - Batch Gradient Descent: {:.6f}'.format(j.squeeze()))
        
    def test_nm(self, X, y):
        X = self._adjust_X(X)
        y = self._adjust_y(y)
        h = self._predict(X, self.theta_nm)
        j = self._mse(y, h)
        print('Test Loss - Newtons Method: {:.6f}'.format(j.squeeze()))
        
    def test_ne(self, X, y):
        X = self._adjust_X(X)
        y = self._adjust_y(y)
        h = self._predict(X, self.theta_ne)
        j = self._mse(y, h)
        print('Test Loss - Normal Equation: {:.6f}'.format(j.squeeze()))
        
        
    def predict_gd(self, X):
        X = self._adjust_X(X)
        h = self._predict(X, self.theta_gd)
        return h
    
    def predict_nm(self, X):
        X = self._adjust_X(X)
        h = self._predict(X, self.theta_nm)
        return h
    
    def predict_ne(self, X):
        X = self._adjust_X(X)
        h = self._predict(X, self.theta_ne)
        return h