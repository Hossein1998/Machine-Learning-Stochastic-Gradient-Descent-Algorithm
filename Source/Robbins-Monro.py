import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random 
from math import exp



df=pd.read_csv("Weekly.csv")
x=df.iloc[0:,1:7]

 
 
x=np.array(x)
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)
 


y=np.array(df[['Direction']])

for data in range(len(y)):
    if(y[data]=='Up'):
        y[data]=1
    else:
         y[data]=0
       
 

 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


lx_train=len(x_train)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

#cost function
def cost(data,params):
    total_cost=0
    for i in range(lx_train):
        total_cost+=(1/lx_train) * ((data[i]*params).sum()-y_train[i])**2
    return total_cost


def sgd(data,params,lrate,epochs,batchsize) :
    
    for i in range(epochs):
        for batch in iterate_minibatches(x_train, y_train, batchsize):
            x_batch, y_batch = batch
            lx_batch=len(x_batch)
        slopes=np.zeros(6)
        beta=random.uniform(0,1)
        v=0.0
        v=beta*v
        alpha=0.1
       
        for j in range(lx_batch):
            for k in range(6):
                slopes[k]+= (1/lx_batch)*((x_batch[j]*params).sum()-  y_batch[j])*x_batch[j][k]
       
        params = params-alpha*(beta*v+lrate*slopes)
        # print(cost(data,params))
    return(params)


def predict(x_test):
    u=random.uniform(0.0, 0.1**2)
    y = 1
    for i in range(len(x_test)):
        y += params[i] * x_test[i]
    return (1.0 / (1.0 + exp(-0.5*y)))-0.5+u

 

params=np.zeros(6)
lrate = 0.01
epochs = 200
batchsize=32
params=sgd(x_train,params,lrate,epochs,batchsize)


# print(params)








