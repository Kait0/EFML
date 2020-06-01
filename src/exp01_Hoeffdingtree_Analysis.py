import shap as sh
import skmultiflow as skm
import sklearn as skl
import numpy as np
import pandas as pd


#-----------------------------------------------------------------------------------------
#Get and clean data
#-----------------------------------------------------------------------------------------
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

train_labels = train_data['50K']
test_labels  = test_data['50K']
train_data   = train_data.drop('50K',1)
test_data    = test_data.drop('50K', 1)

#TODO this is not a particular smart way to deal with categorical data. https://stackoverflow.com/questions/38108832/passing-categorical-data-to-sklearn-decision-tree
#['age', 'workclass', 'finalweight', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '50K
#Turn categorical data into numerical data
le = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['workclass'])
le.fit(attrib)
train_data['workclass'] = le.transform(train_data['workclass'])
test_data['workclass'] = le.transform(test_data['workclass'])

attrib = np.unique(train_data['education'])
le.fit(attrib)
train_data['education'] = le.transform(train_data['education'])
test_data['education'] = le.transform(test_data['education'])

attrib = np.unique(train_data['marital-status'])
le.fit(attrib)
train_data['marital-status'] = le.transform(train_data['marital-status'])
test_data['marital-status'] = le.transform(test_data['marital-status'])

attrib = np.unique(train_data['occupation'])
le.fit(attrib)
train_data['occupation'] = le.transform(train_data['occupation'])
test_data['occupation'] = le.transform(test_data['occupation'])

attrib = np.unique(train_data['relationship'])
le.fit(attrib)
train_data['relationship'] = le.transform(train_data['relationship'])
test_data['relationship'] = le.transform(test_data['relationship'])

attrib = np.unique(train_data['race'])
le.fit(attrib)
train_data['race'] = le.transform(train_data['race'])
test_data['race'] = le.transform(test_data['race'])

attrib = np.unique(train_data['sex'])
le.fit(attrib)
train_data['sex'] = le.transform(train_data['sex'])
test_data['sex'] = le.transform(test_data['sex'])

attrib = np.unique(train_data['native-country'])
le.fit(attrib)
train_data['native-country'] = le.transform(train_data['native-country'])
test_data['native-country'] = le.transform(test_data['native-country'])


#-----------------------------------------------------------------------------------------
# Train Model
#-----------------------------------------------------------------------------------------

tree = skl.tree.DecisionTreeClassifier()
tree.fit(train_data, train_labels)

predicted_labels = tree.predict(test_data)

#-----------------------------------------------------------------------------------------
# Calculate accuracy
#-----------------------------------------------------------------------------------------
def test_accuracy(y_predict_, ys_test_):
    acc = 0.0
    for idx, elem in enumerate(ys_test_):
        if(elem == y_predict_[idx]):
            acc += 1
    acc /= len(y_predict_)
    return acc

accurcy = test_accuracy(predicted_labels, test_labels)

print("Accuracy: ",accurcy)

print("Done :)")