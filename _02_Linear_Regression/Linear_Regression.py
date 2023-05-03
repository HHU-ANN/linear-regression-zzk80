# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y = read_data()
    lambd = -1e-5
    weight = np.dot(np.linalg.inv((np.dot(x.T,x)+np.dot(lambd,np.eye(6)))),np.dot(x.T,y))
    return weight @ data
    
def lasso(data):
    label = 1e-5
    x,y = read_data()
    w = np.zeros([6,1])
    rate = 1e-5
    for i in range(9999999):
        y_predict = np.dot(w, x.T)
        loss = ((y_predict - y) ** 2) / 6
        if loss < label:
            break
        w= w -rate*(y_predict - y)*x.T 
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
