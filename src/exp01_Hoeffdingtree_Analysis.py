import shap as sh
import skmultiflow as sk
import numpy as np
import pandas as pd

data_path="../dat/"
column_names = ['age', 'workclass', 'finalweight', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50K']

train_data = pd.read_csv(filepath_or_buffer=data_path + "adult.data", sep= ", ", names=column_names, engine='python')
test_data  = pd.read_csv(filepath_or_buffer=data_path + "adult.test", sep= ", ", names=column_names, engine='python')

#Deal with ? values by replacing with the mode frequent entry in that column
attrib, counts = np.unique(train_data['workclass'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
train_data['workclass'] = train_data['workclass'].replace({'?': most_freq_attrib})
test_data['workclass']  = test_data['workclass'].replace({'?': most_freq_attrib})

attrib, counts = np.unique(train_data['occupation'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
train_data['occupation'] = train_data['occupation'].replace({'?': most_freq_attrib})
test_data['occupation']  = test_data['occupation'].replace({'?': most_freq_attrib})

attrib, counts = np.unique(train_data['native-country'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
train_data['native-country'] = train_data['native-country'].replace({'?': most_freq_attrib})
test_data['native-country']  = test_data['native-country'].replace({'?': most_freq_attrib})

#Replace 50k by numerical value indicating true or false
train_data['50K'] = train_data['50K'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
test_data['50K']  = test_data['50K'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

#Test to check that we got rid of all ?
#for column in column_names:
#    print(test_data[column][test_data[column] == '?'])

print(test_data.head())




#h_tree = sk.classification.trees.hoeffding_tree.HoeffdingTree()

print("Done :)")