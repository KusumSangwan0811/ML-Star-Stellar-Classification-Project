# -*- coding: utf-8 -*-

!pip install opendatasets scikit-learn

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as ex
sns.set_style('whitegrid')
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib inline

SMALL_SIZE = 10
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150

dataset_url1 = 'https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17?resource=download&select=star_classification.csv'

import opendatasets as od
od.download(dataset_url1)

raw_df = pd.read_csv('stellar-classification-dataset-sdss17/star_classification.csv')
raw_df

raw_df.info()

raw_df.describe()

raw_df.head()

print(raw_df['u'].skew())
raw_df['u'].describe()

raw_df['u'] = np.where(raw_df['u'] <-100.0, raw_df['u'].median(), raw_df['u'])
print(raw_df['u'].skew())

print(raw_df['g'].skew())

print(raw_df['z'].skew())

raw_df['g'] = np.where(raw_df['g'] <-100.0, raw_df['g'].median(),raw_df['g'])
print(raw_df['g'].skew())
raw_df['z'] = np.where(raw_df['z'] <-100.0, raw_df['z'].median(),raw_df['z'])
print(raw_df['z'].skew())

raw_df.describe()

raw_df.head()

raw_df['class'].value_counts()

sns.catplot(x = 'class', kind = 'count', data = raw_df,height=3)

ex.pie(raw_df,names='class',title='Proportion of different classes')

"""The data looks a little imbalanced, but for now lets just consider this and lets trains some models."""

raw_df.columns.values

raw_df.head()

df_excluded_class = raw_df.drop(columns=['class'])

corr_matrix = df_excluded_class.corr()

corr_redshift = corr_matrix["redshift"].sort_values(ascending=False)

print(corr_redshift)

raw_df.drop(['rerun_ID', 'cam_col', 'field_ID'], axis=1, inplace=True)

raw_df.head(10)

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(raw_df[raw_df['class']=='STAR'].redshift, bins = 30, ax = axes[0], kde = False)
ax.set_title('Star')
ax = sns.distplot(raw_df[raw_df['class']=='GALAXY'].redshift, bins = 30, ax = axes[1], kde = False)
ax.set_title('Galaxy')
ax = sns.distplot(raw_df[raw_df['class']=='QSO'].redshift, bins = 30, ax = axes[2], kde = False)
ax = ax.set_title('QSO')

plt.figure(figsize=(8,8))
sns.heatmap(raw_df.corr(),annot=True,linewidths=0.6,fmt=".2f",cmap="coolwarm")
plt.show()

corr = raw_df.corr()
corr["redshift"].sort_values(ascending=False)

raw_df.drop(['run_ID', 'spec_obj_ID', 'plate', 'alpha'], axis=1, inplace=True)

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(raw_df[raw_df['class']=='STAR'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[0], cmap='coolwarm',annot=True)
ax.set_title('Star')
ax = sns.heatmap(raw_df[raw_df['class']=='GALAXY'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[1], cmap='coolwarm',annot=True)
ax.set_title('Galaxy')
ax = sns.heatmap(raw_df[raw_df['class']=='QSO'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[2], cmap='coolwarm',annot=True)
ax = ax.set_title('QSO')

updated_df = raw_df

scaler = MinMaxScaler()
sdss = scaler.fit_transform(updated_df.drop('class', axis=1))

# encoding class labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(updated_df['class'])
updated_df['class'] = y_encoded

X_train, X_test, y_train, y_test = train_test_split(updated_df.drop('class', axis=1), updated_df['class'], test_size=0.25)

knn = KNeighborsClassifier()
training_start = time.perf_counter()
knn.fit(X_train, y_train)
training_end = time.perf_counter()
pred_train = knn.predict(X_train)
acc_knn_train = (pred_train == y_train).sum().astype(float) / len(pred_train)*100
print("K-Nearest Neighbors Classifier's prediction accuracy on training set is: %3.2f" % (acc_knn_train))
prediction_start = time.perf_counter()
preds = knn.predict(X_test)
prediction_end = time.perf_counter()
acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100
knn_train_time = training_end-training_start
knn_prediction_time = prediction_end-prediction_start
print("K-Nearest Neighbors Classifier's prediction accuracy on test set is: %3.2f" % (acc_knn))
print("Time consumed for training: %4.3f seconds" % (knn_train_time))
print("Time consumed for prediction: %4.3f seconds" % (knn_prediction_time))

knn.score(X_test, y_test)

gnb = GaussianNB()
training_start = time.perf_counter()
gnb.fit(X_train, y_train)
training_end = time.perf_counter()
preds_train = gnb.predict(X_train)
acc_gnb_train = (preds_train == y_train).sum().astype(float) / len(preds_train)*100
print("Gaussian Naive Bayes Classifier's prediction accuracy on training set is: %3.2f" % (acc_gnb_train))
prediction_start = time.perf_counter()
preds = gnb.predict(X_test)
prediction_end = time.perf_counter()
acc_gnb = (preds == y_test).sum().astype(float) / len(preds)*100
gnb_train_time = training_end-training_start
gnb_prediction_time = prediction_end-prediction_start
print("Gaussian Naive Bayes Classifier's prediction accuracy on testing set is: %3.2f" % (acc_gnb))
print("Time consumed for training: %4.3f seconds" % (gnb_train_time))
print("Time consumed for prediction: %4.3f seconds" % (gnb_prediction_time))


dtc = DecisionTreeClassifier(random_state=42)
training_start = time.perf_counter()
dtc.fit(X_train, y_train)
training_end = time.perf_counter()
pred_train = dtc.predict(X_train)
acc_dtc_train = (pred_train == y_train).sum().astype(float) / len(pred_train)*100
print("Decision Tree Classifier's prediction accuracy on training set is: %3.2f" % (acc_dtc_train))
prediction_start = time.perf_counter()
preds = dtc.predict(X_test)
prediction_end = time.perf_counter()
acc_dtc = (preds == y_test).sum().astype(float) / len(preds)*100
dtc_train_time = training_end-training_start
dtc_prediction_time = prediction_end-prediction_start
print("Decision Tree Classifier's prediction accuracy on test set is: %3.2f" % (acc_dtc))
print("Time consumed for training: %4.3f seconds" % (dtc_train_time))
print("Time consumed for prediction: %4.3f seconds" % (dtc_prediction_time))


xgb = XGBClassifier(n_estimators=10)
training_start = time.perf_counter()
xgb.fit(X_train, y_train)
training_end = time.perf_counter()
pred_train = xgb.predict(X_train)
acc_xgb_train = (pred_train == y_train).sum().astype(float) / len(pred_train)*100
print("XGBoost's prediction accuracy on training set is: %3.2f" % (acc_xgb_train))
prediction_start = time.perf_counter()
preds = xgb.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy on testing set is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %4.3f seconds" % (xgb_prediction_time))


rfc = RandomForestClassifier(n_estimators=10)
training_start = time.perf_counter()
rfc.fit(X_train, y_train)
training_end = time.perf_counter()
pred_train = rfc.predict(X_train)
acc_rfc_train = (pred_train == y_train).sum().astype(float) / len(pred_train)*100
print("Random Forest Classifier's prediction accuracy on training set is: %3.2f" % (acc_rfc_train))
prediction_start = time.perf_counter()
preds = rfc.predict(X_test)
prediction_end = time.perf_counter()
acc_rfc = (preds == y_test).sum().astype(float) / len(preds)*100
rfc_train_time = training_end-training_start
rfc_prediction_time = prediction_end-prediction_start
print("Random Forest Classifier's prediction accuracy on test set is: %3.2f" % (acc_rfc))
print("Time consumed for training: %4.3f seconds" % (rfc_train_time))
print("Time consumed for prediction: %4.3f seconds" % (rfc_prediction_time))


svc = SVC()
training_start = time.perf_counter()
svc.fit(X_train, y_train)
training_end = time.perf_counter()
preds_train = svc.predict(X_test)
acc_svc_train = (preds_train == y_test).sum().astype(float) / len(preds_train)*100
print("Support Vector Machine Classifier's prediction accuracy of training set is: %3.2f" % (acc_svc_train))
prediction_start = time.perf_counter()
preds = svc.predict(X_test)
prediction_end = time.perf_counter()
acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
svc_train_time = training_end-training_start
svc_prediction_time = prediction_end-prediction_start
print("Support Vector Machine Classifier's prediction accuracy of test set is: %3.2f" % (acc_svc))
print("Time consumed for training: %4.3f seconds" % (svc_train_time))
print("Time consumed for prediction: %4.3f seconds" % (svc_prediction_time))


results = pd.DataFrame({
    'Model': ['KNN', 'Naive Bayes', 'Decision Tree',
              'XGBoost', 'Random Forest','SVC'],
    'Train Score': [acc_knn_train, acc_gnb_train, acc_dtc_train, acc_xgb_train, acc_rfc_train, acc_svc_train],
    'Test Score': [acc_knn, acc_gnb, acc_dtc, acc_xgb, acc_rfc,acc_svc],
    'Runtime Training': [knn_train_time, gnb_train_time, dtc_train_time, xgb_train_time, rfc_train_time, svc_train_time],
    'Runtime Prediction': [knn_prediction_time, gnb_prediction_time, dtc_prediction_time, xgb_prediction_time, rfc_prediction_time, svc_prediction_time]})

result_df = results.sort_values(by='Test Score', ascending=False)
result_df = result_df.set_index('Model')
result_df


model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)

model.classes_


model = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_train, y_train)
model.score(X_test, y_test)

model = DecisionTreeClassifier(max_depth=8, random_state=42).fit(X_train, y_train)
model.score(X_test, y_test)

model = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42)
model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)

model = DecisionTreeClassifier(max_depth=11,max_leaf_nodes=128, random_state=42)
model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)

best_model=model

import pickle

filename = 'best_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(best_model, file)

print(f"Model saved to {filename}")

with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully")
