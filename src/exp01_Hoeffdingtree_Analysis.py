#-----------------------------------------------------------------------------------------
#Imports
#-----------------------------------------------------------------------------------------

import shap as sh
import skmultiflow as skm
import sklearn as skl
import numpy as np
import pandas as pd
import matplotlib as mp


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

le1 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['education'])
le1.fit(attrib)
train_data['education'] = le1.transform(train_data['education'])
test_data['education'] = le1.transform(test_data['education'])

le2 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['marital-status'])
le2.fit(attrib)
train_data['marital-status'] = le2.transform(train_data['marital-status'])
test_data['marital-status'] = le2.transform(test_data['marital-status'])

le3 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['occupation'])
le3.fit(attrib)
train_data['occupation'] = le3.transform(train_data['occupation'])
test_data['occupation'] = le3.transform(test_data['occupation'])

le4 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['relationship'])
le4.fit(attrib)
train_data['relationship'] = le4.transform(train_data['relationship'])
test_data['relationship'] = le4.transform(test_data['relationship'])

le5 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['race'])
le5.fit(attrib)
train_data['race'] = le5.transform(train_data['race'])
test_data['race'] = le5.transform(test_data['race'])

le6 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['sex'])
le6.fit(attrib)
train_data['sex'] = le6.transform(train_data['sex'])
test_data['sex'] = le6.transform(test_data['sex'])

le7 = skl.preprocessing.LabelEncoder()
attrib = np.unique(train_data['native-country'])
le7.fit(attrib)
train_data['native-country'] = le7.transform(train_data['native-country'])
test_data['native-country'] = le7.transform(test_data['native-country'])


#-----------------------------------------------------------------------------------------
# Train Model
#-----------------------------------------------------------------------------------------

tree = skl.tree.DecisionTreeClassifier()
tree.fit(train_data, train_labels)

predicted_train_labels = tree.predict(train_data)
predicted_test_labels  = tree.predict(test_data)

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

train_accuracy = test_accuracy(predicted_train_labels, train_labels)
test_accuracy  = test_accuracy(predicted_test_labels, test_labels)


print("train_accuracy: ", train_accuracy)
print("test_accuracy: ",  test_accuracy)

#-----------------------------------------------------------------------------------------
# Explain Model
#-----------------------------------------------------------------------------------------
explainer = sh.TreeExplainer(tree)
shap_values = explainer.shap_values(test_data)

#-----------------------------------------------------------------------------------------
# Reverse labels so we know what they mean
#-----------------------------------------------------------------------------------------
test_data['workclass']      = le.inverse_transform(test_data['workclass'])
test_data['education']      = le1.inverse_transform(test_data['education'])
test_data['marital-status'] = le2.inverse_transform(test_data['marital-status'])
test_data['occupation']     = le3.inverse_transform(test_data['occupation'])
test_data['relationship']   = le4.inverse_transform(test_data['relationship'])
test_data['race']           = le5.inverse_transform(test_data['race'])
test_data['sex']            = le6.inverse_transform(test_data['sex'])
test_data['native-country'] = le7.inverse_transform(test_data['native-country'])

#-----------------------------------------------------------------------------------------
# Show explanations
#-----------------------------------------------------------------------------------------
sh.force_plot(explainer.expected_value[0], shap_values[0][0], test_data.iloc[0,:], matplotlib=True)

print("Done :)")