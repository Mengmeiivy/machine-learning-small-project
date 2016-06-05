import numpy as np
import sklearn
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt


#transform text files into feature vectors and normalize the feature vectors 
training_file = 'mnist_train.txt'
test_file = 'mnist_test.txt'

training_digit = np.empty(2000)
test_digit = np.empty(1000)
training_feature = np.empty((2000, 784))
test_feature = np.empty((1000, 784))

with open (training_file, 'r') as training_file:
    index = 0
    for row in training_file:
        string = row.split(',')
        training_digit[index] = string[0]
        training_feature[index] = np.array(string[1:])
        index += 1
training_feature = training_feature*2/255 - 1

with open (test_file, 'r') as test_file:
    index = 0
    for row in test_file:
        string = row.split(',')
        test_digit[index] = string[0]
        test_feature[index] = np.array(string[1:])
        index += 1
test_feature = test_feature*2/255 - 1

#multi-class svm training algorithm 
def multi_svm_train(training_digit, training_feature, lambda1):
    weight_vector = np.zeros((10, len(training_feature[0])))
    for i in range (10):
        training_digit_temp = np.empty(len(training_digit))
        for j in range (len(training_digit)):
            if (training_digit[j] == i):
                training_digit_temp[j] = 1
            else:
                training_digit_temp[j] = -1
        update = 0
        for k in range (20):
            for l in range (len(training_digit_temp)):
                update += 1
                step_size = 1.0/(update*lambda1)
                if (training_digit_temp[l]*np.dot(weight_vector[i], training_feature[l]) < 1):
                    weight_vector[i] = (1-step_size*lambda1)*weight_vector[i] + step_size*training_digit_temp[l]*training_feature[l]
                else:
                    weight_vector[i] = (1-step_size*lambda1)*weight_vector[i]
    return weight_vector

#multi-class svm testing algorithm 
def multi_svm_test(weight_vector, test_digit, test_feature):
    error_sum = 0.0
    for i in range (len(test_digit)):
        maximum_score = np.dot(weight_vector[0], test_feature[i])
        maximum_index = 0
        for j in range (1, 10):
            if (np.dot(weight_vector[j], test_feature[i]) > maximum_score):
                maximum_score = np.dot(weight_vector[j], test_feature[i])
                maximum_index = j
        if (maximum_index != test_digit[i]):
            error_sum += 1
    error_rate = error_sum / len(test_digit)
    return error_rate



lambda1 = 2**(-3)

"""
#cross-validation with k=5
error_sum = 0.0
kf = KFold(2000, n_folds = 5)
for train_index, test_index in kf:
    training_feature_current, validation_feature_current = training_feature[train_index], training_feature[test_index]
    training_digit_current, validation_digit_current = training_digit[train_index], training_digit[test_index]
    weight_vector = multi_svm_train(training_digit_current, training_feature_current, lambda1)
    error_rate = multi_svm_test(weight_vector, validation_digit_current, validation_feature_current)
    error_sum += error_rate
average_error_rate = error_sum / 5
print ('Average error rate is', average_error_rate)
"""

"""
#show the plot for cross-validation error against log lambda 
lambda_list = np.array(range(-5, 2))
validation_error = [0.14200000000000002, 0.14550000000000002, 0.1345, 0.14150000000000001, 0.1565, 0.186, 0.21150000000000002]
plt.plot(lambda_list, validation_error)
plt.xlabel('log lambda')
plt.ylabel('cross-validation error')
plt.show()
"""

"""
#computes the test error 
weight_vector = multi_svm_train(training_digit, training_feature, lambda1)
error_rate = multi_svm_test(weight_vector, test_digit, test_feature)
print ('The test error rate is', error_rate)
"""




"""
#compute the test error with parameters at their default settings 
clf = OneVsRestClassifier(SVC())
clf.fit(training_feature, training_digit)
print ('The test error is', 1-clf.score(test_feature, test_digit))
"""

"""
#compute the 10-fold cross-validation error with parameters at their default settings 
clf = OneVsRestClassifier(SVC())
scores = sklearn.cross_validation.cross_val_score(clf, training_feature, training_digit, cv=10)
sum_score= 0.0
for i in range (10):
    sum_score += scores[i]
average_score = sum_score/10
print ('Average validation error is', 1-average_score)
"""

"""
#tuning parameters C and gamma 
clf = OneVsRestClassifier(SVC(C=30, gamma=0.006))
scores = sklearn.cross_validation.cross_val_score(clf, training_feature, training_digit, cv=10)
sum_score= 0.0
for i in range (10):
    sum_score += scores[i]
average_score = sum_score/10
print ('Average validation error is', 1-average_score)
"""

#compute the test error with the parameters at the optimal values that I found
clf = OneVsRestClassifier(SVC(C=30, gamma=0.006))
clf.fit(training_feature, training_digit)
print ('The test error is', 1-clf.score(test_feature, test_digit))




