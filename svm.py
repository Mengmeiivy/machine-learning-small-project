import collections
import numpy as np
import matplotlib.pyplot as plt

#split training data into a training set and a validation set
entire_file = 'spam_train.txt'
line_list = []
line_list2 = []
count= 0;
with open (entire_file, 'r') as text_file:
    for row in text_file:
        if (count < 4000):
            line_list.append(row)
        else:
            line_list2.append(row)
        count += 1
            
training_file = 'spam_training.txt'
validation_file = 'spam_validation.txt'
test_file = 'spam_test.txt'

with open (training_file, 'w') as new_text_file:
    for row in line_list:
        new_text_file.write(row)
with open (validation_file, 'w') as new_text_file2:
    for row in line_list2:
        new_text_file2.write(row)

#build a vocabulary dictionary using words as keys and indices as values
all_word_list = []
with open (training_file, 'r') as process_file:
    for row in process_file:
        temp_word_list = []
        for word in row.split():
            if (not word.isdigit()):
                if (word not in temp_word_list):
                    temp_word_list.append(word)
                    all_word_list.append(word)

ctr = collections.Counter()
vocab = {}
counter = 0;
for word in all_word_list:
    ctr[word] += 1
for word in ctr:
    if (ctr[word] >= 30):
        vocab[word] = counter
        counter += 1

#tranform emails into feature vectors
feature_vector = np.zeros((4000, counter))
spam_vector = np.zeros((4000))
feature_vector2 = np.zeros((1000, counter))
spam_vector2 = np.zeros((1000))
feature_vector_test = np.zeros((1000, counter))
spam_vector_test = np.zeros((1000))

email_num = 0
with open (training_file, 'r') as process_file:
    for row in process_file:
        for word in row.split():
            if (not word.isdigit()):
                if (word in vocab):
                    feature_vector[email_num, vocab[word]] = 1
            else:
                if (word == '1'):
                    spam_vector[email_num] = 1
                else:
                    spam_vector[email_num] = -1
        email_num += 1

email_num = 0
with open (validation_file, 'r') as process_file:
    for row in process_file:
        for word in row.split():
            if (not word.isdigit()):
                if (word in vocab):
                    feature_vector2[email_num, vocab[word]] = 1
            else:
                if (word == '1'):
                    spam_vector2[email_num] = 1
                else:
                    spam_vector2[email_num] = -1
        email_num += 1

feature_vector_total = np.concatenate((feature_vector, feature_vector2))
spam_vector_total = np.concatenate((spam_vector, spam_vector2))

email_num = 0
with open (test_file, 'r') as process_file:
    for row in process_file:
        for word in row.split():
            if (not word.isdigit()):
                if (word in vocab):
                    feature_vector_test[email_num, vocab[word]] = 1
            else:
                if (word == '1'):
                    spam_vector_test[email_num] = 1
                else:
                    spam_vector_test[email_num] = -1
        email_num += 1

#the pegasos svm train algorithm
lambda1 = 2**(-8)
objective_value = []
def pegasos_svm_train(feature_vector, spam_vector, lambda1):
    weight_vector = np.zeros((counter))
    update = 0
    support_vector = 0
    for i in range (20):
        for j in range (len(spam_vector)):
            update += 1
            step_size = 1.0/(update*lambda1)
            if (spam_vector[j]*np.dot(weight_vector, feature_vector[j]) < 1):
                weight_vector = (1-step_size*lambda1)*weight_vector + step_size*spam_vector[j]*feature_vector[j]
            else:
                weight_vector = (1-step_size*lambda1)*weight_vector
        #evalute f(w)
        running_sum = 0.0
        for j in range (len(spam_vector)):
            if (spam_vector[j]*np.dot(weight_vector, feature_vector[j]) < 1):
                running_sum += 1 - spam_vector[j]*np.dot(weight_vector, feature_vector[j])
            if ((i == 19) and (spam_vector[j]*np.dot(weight_vector, feature_vector[j]) <= 1)):
                support_vector += 1
        objective_value.append(0.5*lambda1*(np.linalg.norm(weight_vector))**2 + running_sum/len(spam_vector))
    #print ('Number of support vectors is', support_vector)
    return weight_vector

#train the model
weight_vector = pegasos_svm_train(feature_vector_total, spam_vector_total, lambda1)
#print (objective_value)

"""
#show the plot for the SVM objective
iteration = np.array(range(1, 21))*len(spam_vector)
plt.plot(iteration, objective_value, 'ro')
plt.xlabel('Iteration')
plt.ylabel('SVM Objective')
plt.axis([4000, 80000, 0, 0.2])
plt.show()
"""

def pegasos_svm_test(weight_vector, feature_vector, spam_vector):
    error_sum = 0.0
    hinge_loss_sum = 0.0
    for j in range (len(spam_vector)):
        if (spam_vector[j]*np.dot(weight_vector, feature_vector[j]) < 0):
            error_sum += 1
        if ((np.dot(weight_vector, feature_vector[j]) == 0) and (spam_vector[j] == -1)):
            error_sum += 1
        if (spam_vector[j]*np.dot(weight_vector, feature_vector[j]) < 1):
                hinge_loss_sum += 1 - spam_vector[j]*np.dot(weight_vector, feature_vector[j])
    average_error = error_sum / len(spam_vector)
    average_hinge_loss = hinge_loss_sum / len(spam_vector)
    return (average_error, average_hinge_loss)

#test the model
average_error, average_hinge_loss = pegasos_svm_test(weight_vector, feature_vector_test, spam_vector_test)
print ('Average error and avereage hinge loss is', average_error, average_hinge_loss)

"""
#show the plots
training_error = [0.001, 0.002, 0.00475, 0.006, 0.00925, 0.0155, 0.02275, 0.02825, 0.03875, 0.055, 0.1435]
training_hinge_loss = [0.0062852, 0.0120808, 0.022934, 0.0336786, 0.0500876, 0.0747819, 0.107480125, 0.1545614, 0.21905075625, 0.3122913875, 0.455183010938]
validation_error = [0.018, 0.016, 0.018, 0.019, 0.019, 0.022, 0.022, 0.023, 0.03, 0.051, 0.137]
lambda_list = np.array(range(-9, 2))

plt.plot(lambda_list, training_error)
plt.plot(lambda_list, training_hinge_loss)
plt.plot(lambda_list, validation_error)
plt.legend(['Training error', 'Training hinge loss', 'Validation error'], loc = 'upper left')
plt.show()
"""



