import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math 

df=pd.read_csv("Auto.csv")
 

x=df.iloc[:,1:8]
x["bias"]=1

x=np.array(x)
scaler=MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x)

 

y=np.array(df[['mpg']])

scaler=MinMaxScaler()
scaler.fit(y)
y=scaler.transform(y)
 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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




def adagrad(data,params,epochs,batchsize,fudge_factor = 0.0001,stepsize = 0.001) :
    
    for i in range(epochs):
        for batch in iterate_minibatches(x_train, y_train, batchsize):
            x_batch, y_batch = batch
            lx_batch=len(x_batch)
            slopes=np.zeros(8)
     
        for j in range(lx_batch):
            for k in range(8):
                slopes[k]+= ((x_batch[j]*params).sum()-  y_batch[j])*x_batch[j][k]
                alpha= stepsize/(fudge_factor+math.sqrt(abs(slopes[k])))
                                
        params=params -  stepsize * alpha*slopes
        params=params - alpha* stepsize * math.sqrt(abs(slopes[k]))*slopes
        # print(cost(data,params))
    return(params)


    

#runing Gradiant descent
params=np.zeros(8)
epochs=100
batchsize=32
params=adagrad(x_train,params,epochs,batchsize)

# print(params)



def predict(x_test):
          Y  = (1+(params[0] * x_test[0]) +  (params[1] * x_test[1]) + (params[2] * x_test[2]) + (params[3] * x_test[3]) +  
               (params[4] * x_test[4]) + (params[5] * x_test[5]) + (params[6] * x_test[6]) + (params[7] * x_test[7]))
          
             
          return Y 
y_pred=predict(x_test[0])
mse = np.mean((y_test[0] - y_pred)**2)
# print(mse)       


#polynomial
def predict(x_test, params):
    y=1
    for i in range (len(x_test)-1):
        y += params[i]*(x_test[i] **2)
    return y
          

y_pred=predict(x_test[0] , params)
mse = np.mean((y_test[0] - y_pred)**2)
print(mse)

 