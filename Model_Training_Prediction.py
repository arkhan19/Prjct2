from Data_Preprocessing import *
# Importing Libraries
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10000, random_state = 0, criterion='entropy', verbose=1)
classifier.fit(X_train, Y_train.ravel())


# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = Y_pred.reshape(-1,1)

Accuracy = accuracy_score(Y_test, Y_pred, normalize=False, sample_weight=None)












# #t distribution stochastic neighbor embedding (t-SNE) visualization
# tsne = TSNE(n_components=2, random_state=0)
# x_2d = tsne.fit_transform(X)
# x_train_2d = tsne.fit_transform(X_train)
# x_test_2d = tsne.fit_transform(X_test)

