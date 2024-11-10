import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
import torch.utils.data as Data
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import pennylane as qml
import pytorch_lightning as pl
import math
import time


def data_process(data, window_size, predict_size):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    d_y = scaler1.fit_transform(data.iloc[:,-1:])
    data = scaler2.fit_transform(data.iloc[:,:4])

    data_in = []
    data_out = []
    # range(window_size,len(data)-predict_size+1) range(data.shape[0] - window_size - predict_size + 1)
    for i in range(window_size,len(data)-predict_size+1):
        data_in.append(data[i-window_size:i,0:data.shape[0]])
        data_out.append(d_y[i + predict_size - 1:i + predict_size,0])

    data_in = np.array(data_in)
    data_out = np.array(data_out)

    data_process = {'datain': data_in, 'dataout': data_out}

    return data_process, scaler1 , scaler2


def data_preprocess(src='./Data/city_hour.csv',city='Delhi',datetime='2016-01-01 00:00:00', date_column = 'Datetime', datemore = None,
                    imp_columns=['PM2.5','PM10','CO','AQI'], target='AQI'):
    dfi = pd.read_csv(src)
    dfi.dropna()
    if city is not None:
        dfi = dfi.loc[ dfi['City'] == city]
    if datemore is None:
        dfi = dfi.loc[dfi[date_column] < datetime]
    else: 
        dfi = dfi.loc[dfi[date_column] > datetime]
    dfi = dfi[imp_columns]
    dfi = dfi.reset_index()
    dfi['nxt_target'] = dfi[target].shift(-1)
    dfi['nxt_target'][len(dfi)-1] = dfi['nxt_target'][len(dfi)-2]

    size = int(len(dfi) * 0.8)

    train = dfi.iloc[:size].copy()
    test = dfi.iloc[size:].copy()
    
    features_size = 4
    window_size = 10
    predict_size = 1

    train_processed, train_target_scalar, train_scaler = data_process(train, window_size, predict_size)
    X_train, y_train = train_processed['datain'], train_processed['dataout']

    test_processed, test_target_scalar, test_scaler = data_process(test, window_size, predict_size)
    X_test, y_test = test_processed['datain'], test_processed['dataout']

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))

    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    return X_train, X_test, y_train, y_test, train_target_scalar, train_scaler, test_target_scalar, test_scaler

def runmodel(modelQ, trainDataloader,num_epochs,device,criterion,optimizerQ):
    histQ = np.zeros(num_epochs)
    histQacc = np.zeros(num_epochs)
    count = 0
    for epoch in range(num_epochs):
        loss_Q = []
        rmse_q = []
        correct = 0
        batches = 0
        for (x, y) in trainDataloader:
            modelQ.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = modelQ(x)
            label = y.squeeze(1)
            lossQ = criterion(output, label)
            lossQ.backward()
            optimizerQ.step()
            loss_Q.append(lossQ.item())
        histQ[epoch] = np.sum(loss_Q)
        print(f'[{epoch+1}/{num_epochs}]  LossQ:{np.sum(loss_Q)}')

    return histQ


def viz(histQ,modelQ,X_test,X_train,y_test,y_train,train_target_scalar,test_target_scalar):
    #np.savetxt('./SavedModels/Loss/stacked_qgru.txt',histQ)
    plt.figure(figsize = (12, 6))
    plt.plot(histQ, color = 'blue', label = 'Loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.legend(loc = 'upper right')

    print('\n\n\n')

    pred_y_train = modelQ(X_train)
    pred_y_test = modelQ(X_test)
    pred_y_train = pred_y_train.reshape(-1, 1)
    pred_y_test = pred_y_test.reshape(-1, 1)
    y_train_true = train_target_scalar.inverse_transform(y_train)
    y_train_pred = train_target_scalar.inverse_transform(pred_y_train.cpu().detach().numpy())
    y_test_true = test_target_scalar.inverse_transform(y_test)
    y_test_pred = test_target_scalar.inverse_transform(pred_y_test.cpu().detach().numpy())
    #np.savetxt('./SavedModels/Train/stacked_qgru.txt',y_train_pred)
    #np.savetxt('./SavedModels/Test/stacked_qgru.txt',y_test_pred)


    plt.figure(figsize=(20, 13))
    plt.plot(y_train_true, color = 'red', label = 'Acutal')
    plt.plot(y_train_pred, color = 'blue', label = 'Predict')
    plt.title('Prediction comparison')
    plt.ylabel('Target')
    plt.xlabel('Days')
    plt.legend(loc = 'upper right')
    MSE = mean_squared_error(y_train_true, y_train_pred)
    RMSE = math.sqrt(MSE)
    print(f'Training dataset RMSE:{RMSE}')

    print('\n\n\n')

    plt.figure(figsize=(20, 13))
    plt.plot(y_test_true, color = 'red', label = 'Acutal')
    plt.plot(y_test_pred, color = 'blue', label = 'Predict')
    plt.title('QLSTM prediction comparison')
    plt.ylabel('AQI')
    plt.xlabel('Days')
    plt.legend(loc = 'upper right')

    MSE = mean_squared_error(y_test_true, y_test_pred)
    RMSE = math.sqrt(MSE)
    print(f'Training dataset RMSE:{RMSE}')
    #torch.save(modelQ.state_dict(),'./SavedModels/AQI/stacked_QGRU_aqi_sd')
