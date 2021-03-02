# Imported Libraries
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from tqdm.notebook import tqdm
from time import sleep

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

#loading training datasets
now_path = os.getcwd()
file_path = os.getcwd() + '''training datasets locations'''
print(file_path)
df = pd.read_csv(file_path)
df.head()

#mapping labels
'''
  Argument:
      'Normal'             --> 0
      'Probing-Nmap'       --> 1
      'Probing-Port sweep' --> 2
      'Probing-IP sweep'   --> 3
      'DDOS-smurf'         --> 4
'''
df['label'] = df['label'].map({'Normal':0, 'Probing-Nmap':1, 'Probing-Port sweep':1, 'Probing-IP sweep':1, 'DDOS-smurf':1})
df.head()

df.describe()

#check there's any null
df.isnull().sum().max()
null_num = df.isnull().sum().max()
if null_num == 0:
    print('Good!!! No Null Values!')


#see & plot the features' distribution
feature_field = ['''
        fill the features' name which you want to parse
            ''']
print(len(feature_field))

fig, ax = plt.subplots(10, 2, figsize=(18, 6))
now_feature_value = []
color_set = ['y', 'c', 'm'] 
k=0
idx=0
for i in range(0, 10):
    if idx == 2:
        idx = 0
    else:
        idx += 1
    for j in range(0, 2):
        col_name = str(feature_field[k])
        locals()[str(feature_field[k])+'_val'] = df[col_name].values
        now_feature_value.append(str(feature_field[k])+'_val')
        print("Feature {} : {} -->{} " .format(k ,now_feature_value[k], locals()[str(feature_field[k])+'_val']))
    
        sns.distplot(locals()[str(feature_field[k])+'_val'] , ax=ax[i, j], color='{}'.format(color_set[idx]))
        ax[i, j].set_title("Distribution of {}".format(now_feature_value[k]) , fontsize=14)
        ax[i, j].set_xlim([min(locals()[str(feature_field[k])+'_val']), max(locals()[str(feature_field[k])+'_val'])])
        
        k=k+1
fig.tight_layout()


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print('Normal', round(df['label'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Danger', round(df['label'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('label', axis=1)
y = df['label']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels

# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

df = df.sample(frac=1)

# amount of fraud classes 492 rows.
danger_df = df.loc[df['label'] == 1]
normal_df = df.loc[df['label'] == 0][:1000]

normal_distributed_df = pd.concat([normal_df, danger_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

# plot the features' correlation
# Make sure we use the subsample in our correlation  
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()


