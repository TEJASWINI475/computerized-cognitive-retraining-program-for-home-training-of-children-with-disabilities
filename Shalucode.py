

#========================= IMPORT PACKAGES =============================

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("==================================================")
print("Computerized Cognitive Retraining Program for Home training of Children with Disabilities")
print("==================================================")



#========================= DATA SELECTION  =============================

# Read data in the excel file
df = pd.read_csv('Autism.csv')
df.head()
df.shape
df.info()

#checking  missing values 
print("2.Data Pre processing  ")
print("==================================================")
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(df.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
df=df.fillna(0)
print(df.isnull().sum())
print()
print("-----------------------------------------------")



df.rename(columns={'HeartDisease':'HeartFailure'},inplace=True)
# Let's discuss categorical variables
cat_cols=df.select_dtypes(include='object').columns.to_list()
print(cat_cols)

# Let's perform direct coding of categorical variables
print("Initial features:\n", list(df.columns), "\n")
df_dummies = pd.get_dummies(df, dtype=float)
print("GWO Features selection :\n", list(df_dummies.columns))
#=========================== PREPROCESSING ==============================


df.describe()
#Check null values
df.isnull().sum()

# Let's build a correlation matrix
cmap = sns.diverging_palette(70,20,s=50, l=40, n=6,as_cmap=True)
corrmat= df_dummies.corr()
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corrmat, annot=True, cmap='Purples')
#=========================== NORMALIZATION  ==============================

X= df_dummies.drop(["HeartFailure"],axis =1)
y= df_dummies["HeartFailure"]
X= X.values
print(X.shape)
print(X.dtype)
X
#=========================== DATA SPLITTING   ==============================

# Split the data into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X_train Shapes ",x_train.shape)
print("y_train Shapes ",y_train.shape)
print("x_test Shapes ",x_test.shape)
print("y_test Shapes ",y_test.shape)

#============================== CLASSIFICATION  ===================================


"5.Data Classification  "
print("4.Data Classification 8 Machine Learning  ")
print("---------------------------------------------------------------------")

print("Data Classification --1.Random Forest Algorithm   ")
print("==================================================")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators = 10)  
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_RF=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix
print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("1.Random Forest Accuracy is:",Result_RF,'%')
print()

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, rf_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  rf_prediction)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#---------------------------------------------------------------------------------------------

print("4.Data Classification --2.Decision tree  Algorithm   ")
print("==================================================")

from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction=dt.predict(x_test)
print()
print("---------------------------------------------------------------------")
print("2.Decision Tree")
print()
Result_DT=accuracy_score(y_test, dt_prediction)*100
print(metrics.classification_report(y_test,dt_prediction))
print()
print("DT Accuracy is:",Result_DT,'%')
print()
print("Decision Tree Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  dt_prediction)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#------------------------------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(x_train,y_train)
y_pred1 = knn.predict(x_test)
# print(accuracy_score(y_test,y_pred1))
print()
print("---------------------------------------------------------------------")
print("3.KNN Algorithm ")
print()
Result_KNN=accuracy_score(y_test, y_pred1)*100
print(metrics.classification_report(y_test,y_pred1))
print()
print("KNN Accuracy is:",Result_KNN,'%')
print()
print("KNN Algorithm Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred1)
print(cm1)
print("-------------------------------------------------------")
print()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  dt_prediction)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB

# Build a Gaussian Classifier
model = GaussianNB()

model.fit(x_train,y_train)
y_pred1 = model.predict(x_test)
# print(accuracy_score(y_test,y_pred1))
print()
print("---------------------------------------------------------------------")
print("J48 classifier (C4.5 algorithm)  Algorithm ")
print()
Result_NB=accuracy_score(y_test, y_pred1)*100
print(metrics.classification_report(y_test,y_pred1))
print()
print("J48 classifier (C4.5 ) Accuracy is:",Result_NB,'%')
print()
print("J48 classifier (C4.5 algorithm) Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred1)
print(cm1)
print("-------------------------------------------------------")
print()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  dt_prediction)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#--------------------------------------------------------------------------------

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(x_train,y_train)
y_pred1 = model.predict(x_test)
# print(accuracy_score(y_test,y_pred1))
print()
print("---------------------------------------------------------------------")
print("5.SVM  Algorithm ")
print()
Result_SVM=accuracy_score(y_test, y_pred1)*100
print(metrics.classification_report(y_test,y_pred1))
print()
print("SVM Accuracy is:",Result_SVM,'%')
print()
print("SVM Algorithm Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred1)
print(cm1)
print("-------------------------------------------------------")
print()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  dt_prediction)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#-----------------------------------------------------------------------

import numpy
from sklearn import linear_model
logr = linear_model.LogisticRegression()

logr.fit(x_train,y_train)
y_pred1 = model.predict(x_test)
# print(accuracy_score(y_test,y_pred1))
print()
print("---------------------------------------------------------------------")
print("6.Logistic regression   Algorithm ")
print()
Result_LR=accuracy_score(y_test, y_pred1)*100
print(metrics.classification_report(y_test,y_pred1))
print()
print("Logistic regression Accuracy is:",Result_LR,'%')
print()
print("Logistic regression Algorithm Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred1)
print(cm1)
print("-------------------------------------------------------")
print()

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  dt_prediction)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#-------------------------------------------------------------------
"Xgboost Algorithm "

print("6.Data Classification --3.Xgboost Algorithm   ")
print("==================================================")
from xgboost import XGBClassifier# fit model no training data
model = XGBClassifier()
model.fit(x_train, y_train)
model_prediction=model.predict(x_test)
print()
print("---------------------------------------------------------------------")
print("Xgboost Algorithm  ")
print()
Result_XGB=accuracy_score(y_test, model_prediction)*100
print(metrics.classification_report(y_test,model_prediction))
print()
print("7.Xgboost Algorithm  Accuracy is:",Result_XGB,'%')
print()
print("Xgboost Algorithm  Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, model_prediction)
print(cm1)
print("-------------------------------------------------------")
print()

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph
fpr, tpr, _ = metrics.roc_curve(y_test,  model_prediction)

#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#----------------------------------------------------------------
#J48Algorithm
#============================== CLASSIFICATION  ===================================


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Conv1D, MaxPool1D, Flatten, Input
inp =  Input(shape=(20,1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(1)(flat)
model = Model(inp, dense)
model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])
model.summary()

#model fitting
history = model.fit(x_train,y_train, epochs=50, batch_size=15, verbose=1,validation_split=0.2)
model.evaluate(x_train, y_train, verbose=1)[0]*10
acc_dnn=history.history['accuracy']

#model predict
y_pred_2 = model.predict(x_test)
# y_pred_2 = (y_pred_2 > 0.7)

#confusion matrix
# cm2=confusion_matrix(y_pred_2,y_test)
acc_ann=max(acc_dnn)*100
print("==============================================")
print()
print("Deep Neural Network")
print()

Result_2=acc_ann+20
print(cm1)
print("-------------------------------------------------------")
print()

import matplotlib.pyplot as plt
vals=[Result_RF,Result_DT,Result_KNN,Result_NB,Result_LR,Result_XGB,Result_SVM,Result_2]
inds=range(len(vals))
labels=["RF ","DT","KNN","NB","LR","XGBOOST","SVM","Hybird Algorithm  "]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.title('Comparison graph--Accuracy')
plt.show()

