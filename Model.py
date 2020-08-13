import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt


dataset = pd.read_csv('Financial_Data.csv') 
dataset.columns

dataset = dataset.drop(columns = 'months_employed')
dataset['personal_account_months'] = (dataset.personal_account_m + dataset.personal_account_y*12)

dataset[['personal_account_months',"personal_account_m","personal_account_y"]]

dataset = dataset.drop(columns = ['personal_account_m','personal_account_y'])


dataset = pd.get_dummies(dataset,drop_first  = True)

response= dataset['e_signed']
user = dataset['entry_id']
dataset = dataset.drop(columns = ['e_signed','entry_id'])

#Split Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset,response,test_size = .2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train1 = pd.DataFrame(sc.fit_transform(x_train))
x_test1 = pd.DataFrame(sc.transform(x_test))

x_train1.columns = x_train.columns.values
x_test1.columns = x_test.columns.values

x_train1.index = x_train.index.values
x_test1.index = x_test.index.values

x_train = x_train1
x_test = x_test1

del x_train1, x_test1



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty= 'l1', random_state=0,solver='liblinear')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

results = pd.DataFrame([['Linear Regression(Lasso)', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])



#SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)



#RBF
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)



#KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_results = pd.DataFrame([['KNN (neighbors = 5)', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)

#Naive
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_results = pd.DataFrame([['Naive Bayes', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)


#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

model_results = pd.DataFrame([['Random Forest (n = 100)', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)


#XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


model_results = pd.DataFrame([['XGBoost', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)


#ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units= 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
classifier.add(Dense(units= 9, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units= 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train,y_train, batch_size= 10, nb_epoch=100)


y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)


model_results = pd.DataFrame([['ANN', acc, prec, rec, f1]],
             columns = ["Model", "Accuracy", "Precision", "Recall", "F1"])
results = results.append(model_results,ignore_index = True)


