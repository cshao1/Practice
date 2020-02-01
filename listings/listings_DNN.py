# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:58:54 2020

@author: cshao
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from sklearn.model_selection import KFold # import KFold


class Airbnb_dnn:
    
    def load_data(self):
        self.raw = pd.read_csv('.\listings.csv')
        p= self.raw['price']
        self.raw=self.raw[(p != 0) & (p< np.quantile(p, 0.99))]
        self.raw = self.raw.drop(['id','host_id','name','host_name'], axis=1)
        self.raw =self.raw.fillna(0)
        self.raw = pd.get_dummies(self.raw, columns=['neighbourhood_group','neighbourhood','room_type'])
        
        self.X = self.raw.drop('price', axis=1).values
        self.y = self.raw['price'].values.reshape(-1,1)
        print('')
    
    def build_model(self, X):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=X.shape[1], activation='tanh'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        
    def train(self,X_train, y_train,X_test, y_test):
        self.result = self.model.fit(X_train, y_train,
                          batch_size=100,
                          epochs=500,
                          validation_data=(X_test, y_test))
        self.score = self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self,X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def run_kfold(self):
        n_splits = 10
        kf = KFold(n_splits,random_state=None, shuffle=False) # Define the split - into 2 folds 
        
        rsArr=[]
        errArr=[]
        self.build_model(self.X)
        
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index,:], self.X[test_index,:]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            minmax_sc_x = MinMaxScaler()
            minmax_sc_y = MinMaxScaler()
            X_train = minmax_sc_x.fit_transform(X_train)
            y_train = minmax_sc_y.fit_transform(y_train)
            X_test = minmax_sc_x.transform(X_test)
            y_real = y_test
            y_test = minmax_sc_y.transform(y_test) 
            
            self.train(X_train, y_train, X_test, y_test)
            y_est = self.predict(X_test)
            y_pred = minmax_sc_y.inverse_transform(y_est)
            
            err = np.abs(y_pred - y_real)
            mse = np.sqrt(np.mean(err**2))
            mae = np.mean(np.mean(err))
            errArr.append([mse,mae])
            rsArr.append(np.hstack((y_pred, y_real)))
        return errArr, rsArr
            
if __name__ == '__main__':
    dnn = Airbnb_dnn()
    dnn.load_data()
    errArr, rsArr = dnn.run_kfold()
    errArrD = np.array(errArr)