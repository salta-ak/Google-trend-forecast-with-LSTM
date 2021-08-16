


  #building LSTM model 
model = Sequential()
model.add(LSTM(1, input_shape=(1, look_back), return_sequences=True))
model.add(Dense(1)) 
print("\n -------------------------------- \n Model Summary \n-------------------------------- \n ")
print(model.summary())

model.compile(loss='mean_squared_error', optimizer='adam')
path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)
modelckpt_callback = keras.callbacks.ModelCheckpoint( monitor="val_loss",filepath=path_checkpoint,  verbose=1,  save_weights_only=True, save_best_only=True)
es_callback = CSVLogger('log.csv', append=True, separator=';')


r = model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=2,callbacks=[es_callback, modelckpt_callback,])

# making predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
    
# inverse transforming results
train_predict, y_train, test_predict, y_test = inverse_transforms(train_predict, y_train, test_predict, y_test, df.bitcoin, train_dates, test_dates, scaler, [True])
print("\n -------------------------------- \n RMSE \n -------------------------------- \n")
error = np.sqrt(mean_squared_error(train_predict, y_train))
print('Train RMSE: %.3f' % error)
error = np.sqrt(mean_squared_error(test_predict, y_test))
print('Test RMSE: %.3f' % error)
path_checkpoint = "model_checkpoint.h5"
