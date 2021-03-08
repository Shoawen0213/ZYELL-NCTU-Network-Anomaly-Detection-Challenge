#!/usr/bin/env python
# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier



os.getcwd()



def valid_modell(model, x_test, y_test):
    _predict = model.predict(x_test)
    print("shape of _predict : ", _predict.shape)
    unique, counts = np.unique(_predict, return_counts=True)
    print("_predict contains:{}".format(dict(zip(unique, counts))))

    #run score(define by shao_pun)
    get_ipython().run_line_magic('run', 'score.py')

    #transform into numpy array
    y_array = np.array(y_test)
    _pred_array = np.array(_predict).reshape(len(_predict), 1)

    score_ = score(_pred_array, y_array)
    print('The score that calculated by SP version of XGBClassifier on testing data set', score_)


# In[4]:


file_path = ' loading trainind data path'
training_data_X_pred_prob_7XGB = pd.read_csv(file_path)
training_data_X_pred_prob_7XGB     


file_path = 'loading trainind_label data path'
y_label = pd.read_csv(file_path)
y_label     


y_label['label'].value_counts()


#split data
random_state = random.randint(0, 300)
print("Random seed this time : ", random_state)
x_train, x_test, y_train, y_test = train_test_split(training_data_X_pred_prob_7XGB, y_label, test_size=0.3, random_state=random_state)
'''
  Argument:
      'Normal'             --> 1
      'Probing-Nmap'       --> 3
      'Probing-Port sweep' --> 4
      'Probing-IP sweep'   --> 2
      'DDOS-smurf'         --> 0
'''

label_list = ['Normal', 'DOS-smurf', 'Probing-Ipsweep', 'Probing-Nmap', 'Probing-Port sweep']
print("label_list : {}".format(label_list))
print("//--------------------------------------------//")

#label encode
labelencoder = LabelEncoder()
y_train_LE = labelencoder.fit_transform(y_train)
print("y_train labeled after label encode : {}".format(y_train_LE))

print("shape of y_train data : ", y_train_LE.shape)
unique, counts = np.unique(y_train_LE, return_counts=True)
print("y_train data contains:{} ".format(dict(zip(unique, counts))))

print("//--------------------------------------------//")

y_test_LE = labelencoder.fit_transform(y_test)
print("y_test labeled after label encode : {}".format(y_test_LE))

print("shape of y_train_LE data : ", y_test_LE.shape)
unique, counts = np.unique(y_test_LE, return_counts=True)
print("y_test_LE data contains:{} ".format(dict(zip(unique, counts))))


# In[8]:


print("x_train shape is: {}".format(x_train.shape))
print("x_test shape  is: {}".format(x_test.shape))
print("y_train shape is: {}".format(y_train.shape))
print("y_test shape  is: {}".format(y_test.shape))


# In[9]:


y_train_LE


# In[10]:


clf1 = tree.DecisionTreeClassifier(max_depth=12)
dt_clf = clf1.fit(x_train, y_train_LE)
predicted_y = dt_clf.predict(x_test)
print("Accurancy on x_test is :", accuracy_score(y_test, predicted_y))
print('Accurancy on x_test by using "score API" is :', dt_clf.score(x_test, y_test_LE))

#save model
joblib.dump(dt_clf, "path for save model")


#load model
model = joblib.load('path for save model')

valid_modell(model, x_test, y_test_LE)


# In[ ]:





# In[11]:


clf2 = MLPClassifier(solver='sgd', activation='relu')
mlp_clf = clf2.fit(x_train, y_train_LE)
predicted_y = mlp_clf.predict(x_test)
print("Accurancy on x_test is :", accuracy_score(y_test, predicted_y))
print('Accurancy on x_test by using "score API" is :', mlp_clf.score(x_test, y_test_LE))

#save model
joblib.dump(mlp_clf, "path for save model")


#load model
model = joblib.load('path for save model')

valid_modell(model, x_test, y_test_LE)


random_state = random.randint(0, 250)
clf3 = RandomForestClassifier(n_estimators=100, random_state=random_state,
                              verbose=1, n_jobs=-1)
rf_clf = clf3.fit(x_train, y_train_LE)
predicted_y = rf_clf.predict(x_test)
print("Accurancy on x_test is :", accuracy_score(y_test,predicted_y))
print('Accurancy on x_test by using "score API" is :', mlp_clf.score(x_test, y_test_LE))

#save model
joblib.dump(rf_clf, "path for save model")


#load model
model = joblib.load('path for save model')

valid_modell(model, x_test, y_test_LE)




file_path = 'loading training_data path'
training_data_X_pred_prob_7RFC = pd.read_csv(file_path)
training_data_X_pred_prob_7RFC     #9241463 rows × 15 columns


# In[14]:


file_path = 'loading training_data label path'
y_label = pd.read_csv(file_path)
y_label     #9241463 rows × 15 columns


# In[15]:


#split data
random_state = random.randint(0, 300)
print("Random seed this time : ", random_state)
x_train, x_test, y_train, y_test = train_test_split(training_data_X_pred_prob_7RFC, y_label, test_size=0.3, random_state=random_state)
'''
  Argument:
      'Normal'             --> 1
      'Probing-Nmap'       --> 3
      'Probing-Port sweep' --> 4
      'Probing-IP sweep'   --> 2
      'DDOS-smurf'         --> 0
'''

label_list = ['Normal', 'DOS-smurf', 'Probing-Ipsweep', 'Probing-Nmap', 'Probing-Port sweep']
print("label_list : {}".format(label_list))
print("//--------------------------------------------//")

#label encode
labelencoder = LabelEncoder()
y_train_LE = labelencoder.fit_transform(y_train)
print("y_train labeled after label encode : {}".format(y_train_LE))

print("shape of y_train data : ", y_train_LE.shape)
unique, counts = np.unique(y_train_LE, return_counts=True)
print("y_train data contains:{} ".format(dict(zip(unique, counts))))

print("//--------------------------------------------//")

y_test_LE = labelencoder.fit_transform(y_test)
print("y_test labeled after label encode : {}".format(y_test_LE))

print("shape of y_train_LE data : ", y_test_LE.shape)
unique, counts = np.unique(y_test_LE, return_counts=True)
print("y_test_LE data contains:{} ".format(dict(zip(unique, counts))))


# In[16]:


clf1 = tree.DecisionTreeClassifier(max_depth=12)
dt_clf = clf1.fit(x_train, y_train_LE)
predicted_y = dt_clf.predict(x_test)
print("Accurancy on x_test is :", accuracy_score(y_test, predicted_y))
print('Accurancy on x_test by using "score API" is :', dt_clf.score(x_test, y_test_LE))

#save model
joblib.dump(dt_clf, "path for save model")


#load model
model = joblib.load('path for save model')

valid_modell(model, x_test, y_test_LE)


# In[17]:


clf2 = MLPClassifier(solver='sgd', activation='relu')
mlp_clf = clf2.fit(x_train, y_train_LE)
predicted_y = mlp_clf.predict(x_test)
print("Accurancy on x_test is :", accuracy_score(y_test, predicted_y))
print('Accurancy on x_test by using "score API" is :', mlp_clf.score(x_test, y_test_LE))

#save model
joblib.dump(mlp_clf, "path for save model")


#load model
model = joblib.load('path for save model')

valid_modell(model, x_test, y_test_LE)


# In[18]:


random_state = random.randint(0, 250)
clf3 = RandomForestClassifier(n_estimators=100, random_state=random_state,
                              verbose=1, n_jobs=-1)
rf_clf = clf3.fit(x_train, y_train_LE)
predicted_y = rf_clf.predict(x_test)
print("Accurancy on x_test is :", accuracy_score(y_test,predicted_y))
print('Accurancy on x_test by using "score API" is :', mlp_clf.score(x_test, y_test_LE))

#save model
joblib.dump(rf_clf, "path for save model")


#load model
model = joblib.load('path for save model')

valid_modell(model, x_test, y_test_LE)

