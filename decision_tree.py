import numpy as np
import re
from itertools import chain
import sklearn
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import os

training_file = 'adult_train.txt'
test_file = 'adult_test.txt'
feature_file = 'features.txt'

"""
# find the length of the training and test file
training_num = 0
with open (training_file, 'r') as training_file:
    for row in training_file:
        training_num += 1

test_num = 0
with open (test_file, 'r') as test_file:
    for row in test_file:
        test_num += 1
"""

training_num = 32561
test_num = 16281

training_feature = np.empty((training_num, 12), dtype = object)
training_label = np.zeros((training_num))
test_feature = np.empty((test_num, 12), dtype = object)
test_label = np.zeros((test_num))

sum_age = 0.0
num_age = 0.0
sum_gain = 0.0
num_gain = 0.0
sum_loss = 0.0
num_loss = 0.0
sum_hours = 0.0
num_hours= 0.0

# build the feature and label vectors for training file
with open (training_file, 'r') as training_file:
    index = 0
    for row in training_file:
        string = re.split(r'[, ]+', row)
        if (string[0].isdigit):
            sum_age += float(string[0])
            num_age += 1
        if (string[8].isdigit):
            sum_gain += float(string[8])
            num_gain += 1
        if (string[9].isdigit):
            sum_loss += float(string[9])
            num_loss += 1
        if (string[10].isdigit):
            sum_hours += float(string[10])
            num_hours += 1
        training_feature[index] = np.array(string[:12])
        string[12] = string[12].strip()
        if (string[12] == '>50K'):
            training_label[index] = 1
        else:
            training_label[index] = -1
        index += 1

# make the substitution list 
sub_list = [0]*12

sub_list[0] = sum_age / num_age
sub_list[8] = sum_gain / num_gain
sub_list[9] = sum_loss / num_loss
sub_list[10] = sum_hours / num_hours

for i in chain(range (1, 8), range (11, 12)):
    unique,pos = np.unique(training_feature[:, i], return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    sub_list[i] = unique[maxpos]

# substitute the missing values in training vectors
for i in range (training_num):
    for j in range (12):
        if (training_feature[i][j] == '?'):
            training_feature[i][j] = sub_list[j]

# build the feature and label vectors for test file
with open (test_file, 'r') as test_file:
    index = 0
    for row in test_file:
        string = re.split(r'[, ]+', row)
        test_feature[index] = np.array(string[:12])
        string[12] = string[12].strip()
        if (string[12] == '>50K'):
            test_label[index] = 1
        else:
            test_label[index] = -1
        index += 1       

# substitute the missing values in test vectors
for i in range (test_num):
     for j in range (12):
        if (test_feature[i][j] == '?'):
            test_feature[i][j] = sub_list[j]

# create a new feature list
new_feature = []
with open (feature_file, 'r') as feature_file:
    index = 0
    for row in feature_file:
        string = re.split(r'[:, ]+', row)
        string[len(string)-1] = string[len(string)-1].replace('.\n', '')
        string[len(string)-1] = string[len(string)-1].replace('.', '')
        if (index == 0 or index == 8 or index == 9 or index == 10):
            new_feature.append(string[0])
        else:
            for i in range (1, len(string)):
                new_feature.append(string[i])
        index += 1

# build new feature vectors for training and test files
new_feature_length = len(new_feature)
new_training_feature = np.zeros((training_num, new_feature_length))
new_test_feature = np.zeros((test_num, new_feature_length))

for i in range (training_num):
    for j in range (12):
        if (j == 0):
            feature_index = new_feature.index('age')
            new_training_feature[i][feature_index] = training_feature[i][j]
        elif (j == 8):
            feature_index = new_feature.index('capital-gain')
            new_training_feature[i][feature_index] = training_feature[i][j]
        elif (j == 9):
            feature_index = new_feature.index('capital-loss')
            new_training_feature[i][feature_index] = training_feature[i][j]
        elif (j == 10):
            feature_index = new_feature.index('hours-per-week')
            new_training_feature[i][feature_index] = training_feature[i][j]
        else:
            feature_index = new_feature.index(training_feature[i][j])
            new_training_feature[i][feature_index] = 1

for i in range (test_num):
    for j in range (12):
        if (j == 0):
            feature_index = new_feature.index('age')
            new_test_feature[i][feature_index] = test_feature[i][j]
        elif (j == 8):
            feature_index = new_feature.index('capital-gain')
            new_test_feature[i][feature_index] = test_feature[i][j]
        elif (j == 9):
            feature_index = new_feature.index('capital-loss')
            new_test_feature[i][feature_index] = test_feature[i][j]
        elif (j == 10):
            feature_index = new_feature.index('hours-per-week')
            new_test_feature[i][feature_index] = test_feature[i][j]
        else:
            feature_index = new_feature.index(test_feature[i][j])
            new_test_feature[i][feature_index] = 1

# shuffle and split the training vectors into training set and validation set
random_state = np.random.get_state()
np.random.shuffle(new_training_feature)
np.random.set_state(random_state)
np.random.shuffle(training_label)

cut_point = int (training_num*0.7)
train_feature, validation_feature = new_training_feature[:cut_point,:], new_training_feature[cut_point:,:]
train_label, validation_label = training_label[:cut_point], training_label[cut_point:]

# scikit learn
# find the best max_depth
depth_train_accuracy = []
depth_validation_accuracy = []
for i in range (30):
    clf = tree.DecisionTreeClassifier(max_depth = i+1)
    clf.fit(train_feature, train_label)
    depth_train_accuracy.append(clf.score(train_feature, train_label))
    depth_validation_accuracy.append(clf.score(validation_feature, validation_label))
best_max_depth = depth_validation_accuracy.index(max(depth_validation_accuracy)) + 1

# find the best min_samples_leaf
leaf_train_accuracy = []
leaf_validation_accuracy = []
for j in range (50):
    clf = tree.DecisionTreeClassifier(min_samples_leaf = j+1)
    clf.fit(train_feature, train_label)
    leaf_train_accuracy.append(clf.score(train_feature, train_label))
    leaf_validation_accuracy.append(clf.score(validation_feature, validation_label))
best_leaf = leaf_validation_accuracy.index(max(leaf_validation_accuracy)) + 1

# draw the top 3 levels of the tree
print ('The best max_depth and min_samples_leaf are', best_max_depth, best_leaf) #10, 43
clf = tree.DecisionTreeClassifier(max_depth = best_max_depth, min_samples_leaf = best_leaf)
clf.fit(train_feature, train_label)
class_names1 = ['<=50K', '>50K']
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file = f, max_depth = 3, feature_names = new_feature, class_names = class_names1, filled = True, rounded = True)

# test the model
clf = tree.DecisionTreeClassifier(max_depth = best_max_depth, min_samples_leaf = best_leaf)
clf.fit(new_training_feature, training_label)
print ('Accuracy on test data is', clf.score(new_test_feature, test_label)) #0.82169399914

# plots 
max_depth = np.array(range(1,31))
plt.plot(max_depth, depth_train_accuracy)
plt.plot(max_depth, depth_validation_accuracy)
plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc = 'upper left')
plt.show()

min_samples_leaf = np.array(range(1,51))
plt.plot(min_samples_leaf, leaf_train_accuracy)
plt.plot(min_samples_leaf, leaf_validation_accuracy)
plt.legend(['Accuracy on training set', 'Accuracy on validation set'], loc = 'upper right')
plt.show()



