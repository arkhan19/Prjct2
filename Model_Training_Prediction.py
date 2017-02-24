# Multiple Regression Model Many Independent Variable
from Data_Preprocessing import *
# Importing Libraries
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
from sklearn.metrics import confusion_matrix
import sklearn.manifold

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0, criterion='entropy', verbose=1)
classifier.fit(X_train, Y_train.ravel())


# Predicting the Test set results
Y_pred = classifier.predict(X_test)

Y_pred = Y_pred.reshape(-1,1)

# Making the Confusion Matrix
# cm = confusion_matrix(Y_test, Y_pred)
#
# plt.scatter(X[:,0],Y, color= 'black')
# plt.plot(X[], Y_pred, color='red')
# plt.title('TITLE')
# plt.xlabel('Factors')
# plt.ylabel('Rating')
# plt.show()