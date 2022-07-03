import tensorflow as tf 
import sklearn
import pandas as pd
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from keras.optimizers import Adam
from matplotlib import pyplot as plt




dataset=pd.read_csv('life_expectency.csv')
dataset=dataset.drop(['Country'],axis=1)

labels=dataset.iloc[:,-1]
features=dataset.iloc[:, 0:-1]

features=pd.get_dummies(features)

features_train, features_test, labels_train, labels_test=train_test_split(features,labels,test_size=0.2,random_state=23)

numerical_features=features.select_dtypes(include=['float64','int64'])
numerical_columns=numerical_features.columns

ct=ColumnTransformer([("onlynumeric", StandardScaler(),numerical_columns)], remainder='passthrough')

features_train_scaled=ct.fit_transform(features_train)

feature_test_scaled=ct.fit_transform(features_test)

my_model=Sequential()

input=InputLayer(input_shape= (features.shape[1], ))

my_model.add(input)

my_model.add(Dense(128, activation='relu'))

my_model.add(Dense(1))

opt=Adam(learning_rate=0.03)

my_model.compile(optimizer=opt, loss='mse', metrics= ['mae'])


my_model.fit(features_train, labels_train, epochs=40, batch_size=1, verbose=1)


res_mse, res_mae=my_model.evaluate(feature_test_scaled,labels_test,verbose=0)



print("the final result of mse is: ",res_mse)
print("the final result of mae is: ",res_mae)

print(my_model.summary())