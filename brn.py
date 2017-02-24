#Using Tensorflow-cpu
from Data_Preprocessing import *
import keras
from keras.models import Sequential #initialise neural network
from keras.layers import Dense  #building of layers


# Initializer for ANN
classifier = Sequential()

# Adding I/P - Hidden - O/P layers
classifier.add(Dense(output_dim=55, init='uniform', activation='relu', input_dim= 110))
classifier.add(Dense(output_dim=55, init='uniform', activation='relu'))  # No Input Dim for 2nd Hidden layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))  # Output Layer; Sigmoid function will
                                                                            # produce probablites for Output layer

# ANN Compilation
classifier.compile(optimizer='rmsprop', metrics=['accuracy'], loss ='sparse_categorical_crossentropy')

#Fitting
classifier.fit(X_train, Y_train, batch_size= 19, nb_epoch=99, shuffle=True)

Y_pred = classifier.predict(X_test)