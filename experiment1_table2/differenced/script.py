
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from keras_tuner import RandomSearch
from keras_tuner import BayesianOptimization
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from keras.callbacks import CSVLogger
import sys
import csv
import sys

df = pd.read_csv('bitcoin.csv', sep=',')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True) 


def create_dataset(data_series, look_back, split_frac, transforms):
    
    # log transforming that data, if necessary
    
    # differencing data, if necessary
    if transforms[0] == True:
        dates = data_series.index
        data_series = pd.Series(data_series - data_series.shift(1), index=dates).dropna()

    # scaling values between 0 and 1
    dates = data_series.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
    data_series = pd.Series(scaled_data[:, 0], index=dates)
    
    # creating targets and features by shifting values by 'i' number of time periods
    df = pd.DataFrame()
    for i in range(look_back+1):
        label = ''.join(['t-', str(i)])
        df[label] = data_series.shift(i)
    df = df.dropna()
    print(df.tail())
    
    # splitting data into train and test sets
    size = int(split_frac*df.shape[0])
    train = df[:size]
    test = df[size:]
    
    # creating target and features for training set
    X_train = train.iloc[:, 1:].values
    y_train = train.iloc[:, 0].values
    train_dates = train.index
    
    # creating target and features for test set
    X_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    test_dates = test.index
    
    # reshaping data into 3 dimensions for modeling with the LSTM neural net
    X_train = np.reshape(X_train, (X_train.shape[0], 1, look_back))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, look_back))
    
    return X_train, y_train, X_test, y_test, train_dates, test_dates, scaler
    
    
    
def inverse_transforms(train_predict, y_train, test_predict, y_test, data_series, train_dates, test_dates, scaler, transforms):
    
    # inverse 0 to 1 scaling
    train_predict = pd.Series(scaler.inverse_transform(train_predict.reshape(-1,1))[:,0], index=train_dates)
    y_train = pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1))[:,0], index=train_dates)

    test_predict = pd.Series(scaler.inverse_transform(test_predict.reshape(-1, 1))[:,0], index=test_dates)
    y_test = pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1))[:,0], index=test_dates)
    if transforms[0] == True:
        train_predict = pd.Series(train_predict + data_series.shift(1), index=train_dates).dropna()
        y_train = pd.Series(y_train + data_series.shift(1), index=train_dates).dropna()

        test_predict = pd.Series(test_predict + data_series.shift(1), index=test_dates).dropna()
        y_test = pd.Series(y_test + data_series.shift(1), index=test_dates).dropna()
        
    return train_predict, y_train, test_predict, y_test
  
  
#sys.stdout = open("output.txt", "w")
look_back=(12,24)
  

    

  #building LSTM model 
for column in df: 
  for window_size in look_back:
    for neuron in (4,8,16,32,60):
      print("\n -------------------------------- \n Training Data \n -------------------------------- \n ")
      X_train, y_train, X_test, y_test, train_dates, test_dates, scaler = create_dataset(df[column], window_size, 0.8, [True])
      model = Sequential()
      model.add(LSTM(neuron, input_shape=(1, window_size)))
      model.add(Dense(1))
      print("\n -------------------------------- \n Model Summary \n-------------------------------- \n ")
      print(model.summary())

      model.compile(loss='mean_squared_error', optimizer='adam')
      path_checkpoint = column+"_"+window_size+"_"+neuron+"model_checkpoint.h5"
      es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
      modelckpt_callback = keras.callbacks.ModelCheckpoint( monitor="val_loss",filepath=path_checkpoint,  verbose=1,  save_weights_only=True, save_best_only=True)
      es_callback = CSVLogger(str(column)+"_"+str(window_size)+"_"+str(neuron)+'log.csv', append=True, separator=';')


      model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2,callbacks=[es_callback, modelckpt_callback,])

# making predictions
      train_predict = model.predict(X_train)
      test_predict = model.predict(X_test)
    
# inverse transforming results
      inverse_transforms(train_predict, y_train, test_predict, y_test, df[column], train_dates, test_dates, scaler,[True])
      print("\n -------------------------------- \n RMSE \n -------------------------------- \n")
      error = np.sqrt(mean_squared_error(train_predict, y_train))
      print('Train RMSE: %.3f' % error)
      error = np.sqrt(mean_squared_error(test_predict, y_test))
      print('Train RMSE: %.3f' % error)
      print('key_word: ' + str(column))
      print('window size: '+ str(window_size))
      print('N neurons: ' + str(neuron )) 
#sys.stdout.close()
