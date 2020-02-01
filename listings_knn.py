# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:57:42 2020

@author: llyyue
"""

import numpy as np
import pandas as pd

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


raw = pd.read_csv('.\listings.csv')
raw = raw.drop(['id','host_id'], axis=1)


p= raw['price']
data=raw[(p != 0) & (p< np.quantile(p, 0.99))]
#data['location']=tuple(zip(data['latitude'],data['longitude']))
#x1= data['latitude']
#x2= data['longitude']
#y1 = data['price']



t=data.nlargest(10, 'price')

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.scatter3D(x, y, z, c=z)

#plt.hist(y, bins=100)
#y1.hist(bins=100)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
cols = ['name',	'host_name','neighbourhood_group','neighbourhood','room_type']
for col in cols:
    data.loc[:,col] =le.fit_transform(data.loc[:,col].replace(np.nan,''))


data = data.fillna(0)
#X=data.drop('price', axis=1)
X=data[['latitude','longitude']]
y=data['price'] 

#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#X=X.drop(['id','host_id','latitude','longitude'],axis=1)
##apply SelectKBest class to extract top 10 best features
#bestfeatures = SelectKBest(score_func=chi2, k=3)
#fit = bestfeatures.fit(X,y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
##concat two dataframes for better visualization 
#featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#print(featureScores.nlargest(10,'Score'))  #print 10 best features



#from sklearn.linear_model import LinearRegression
#model = LinearRegression()
#reg = LinearRegression().fit(X, y)
#reg.score(X, y)
#
#rs=[]
#for icol in range(X.shape[1]):
#    rs.append((X.columns[icol] ,np.corrcoef(X.iloc[:,icol],y)[0,1]))

#X = X.drop('location', axis=1)
#
#
#from sklearn import neighbors
#n_neighbors = 5
#
#rs=[]
#for i, weights in enumerate(['uniform', 'distance']):
#    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
#    knn = knn.fit(X, y)
#    rs.append( knn.predict(X))
#
#debug=np.vstack([rs[0], rs[1], y]).T
#
#from sklearn.neighbors import NearestNeighbors
#neigh = NearestNeighbors(n_neighbors=2)
#neigh.fit(X)
#
#A = neigh.kneighbors_graph(X)
#A.toarray()
    

from sklearn import neighbors
n_neighbors = 5

rs=[]
mse=[]
mae=[]
weights = ['uniform', 'distance']


from sklearn.model_selection import KFold # import KFold
n_splits = 10
kf = KFold(n_splits,random_state=None, shuffle=False) # Define the split - into 2 folds 

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    knn = knn.fit(X_train, y_train)
    y_est = knn.predict(X_test)
    rs.append(tuple(zip(y_est, y_test)))
    mse.append( np.mean((y_est - y_test)**2))
    mae.append( np.mean(np.abs(y_est - y_test)))