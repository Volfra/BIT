import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('BitStudentDataSet.csv')
#print(dataset)
X = pd.DataFrame(dataset.iloc[:,5:8])
Y = pd.DataFrame(dataset.iloc[:,-3])
Y_log = pd.DataFrame(dataset.iloc[:,-2])
#print(X)
#print(Y)
#print(Y_log)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.7, random_state=7)

#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

print('LINEAR REGRESSION')
reg = LinearRegression().fit(X_train, Y_train)

v = pd.DataFrame(reg.coef_, index=['Coeff']).transpose()
w = pd.DataFrame(X.columns, columns=['At'])

coeff_df = pd.concat([w,v], axis=1, join='inner')
print(coeff_df)

y_pred = reg.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['Pred NOTA 4'])
print(y_pred)

print('MeanAbsError', metrics.mean_absolute_error(Y_test, y_pred))
print('MeanSqrError', metrics.mean_squared_error(Y_test, y_pred))
print('RootMeanSqrError', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

print('\nLOGISTIC REGRESSION')

X_train, X_test, Y_train, Y_test = train_test_split(X,Y_log, train_size=0.7, random_state=7)

reg = LogisticRegression().fit(X_train, Y_train.values.ravel())

v = pd.DataFrame(reg.coef_, index=['Coeff']).transpose()
w = pd.DataFrame(X.columns, columns=['At'])

coeff_df = pd.concat([w,v], axis=1, join='inner')
print(coeff_df)

y_pred = reg.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['Pred NOTA 4'])
print(y_pred)

print('MeanAbsError', metrics.mean_absolute_error(Y_test, y_pred))
print('MeanSqrError', metrics.mean_squared_error(Y_test, y_pred))
print('RootMeanSqrError', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

print('\nK NEIGHBORS')

knei = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2).fit(X_train ,Y_train.values.ravel())
y_pred = knei.predict(X_test)
matriz = confusion_matrix(Y_test, y_pred)
print('Confusion Matrix:')
print(matriz)
print('Score ', precision_score(Y_test, y_pred))
