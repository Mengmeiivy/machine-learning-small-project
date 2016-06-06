
def randIndex(truth, predicted):
	"""
	The function is to measure similarity between two label assignments
	truth: ground truth labels for the dataset (1 x 1496)
	predicted: predicted labels (1 x 1496)
	"""
	if len(truth) != len(predicted):
		print ("different sizes of the label assignments")
		return -1
	elif (len(truth) == 1):
		return 1
	sizeLabel = len(truth)
	agree_same = 0
	disagree_same = 0
	count = 0
	for i in range(sizeLabel-1):
		for j in range(i+1,sizeLabel):
			if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
				agree_same += 1
			elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
				disagree_same +=1
			count += 1
	return (agree_same+disagree_same)/float(count)


import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import numpy as np
import pylab as pl
import re
import matplotlib.pyplot as plt
import random

"""
# Plot dendogram and cut the tree to find resulting clusters
#fig = pl.figure() 
plt.figure()
data = np.array([[1,2,3],[1,1,1],[5,5,5]])
datalable = ['first','second','third']
hClsMat = sch.linkage(data, method='complete') # Complete clustering
#sch.dendrogram(hClsMat, labels= datalable, leaf_rotation = 45)
#fig.show()
plt.show()
resultingClusters = sch.fcluster(hClsMat,t= 3, criterion = 'distance')
print (resultingClusters)
"""


# 1. 
# Scaling min max 
file1 = 'dataCereal-grains-pasta.txt'
file2 = 'dataFinfish-shellfish.txt'
file3 = 'dataVegetables.txt'
file4 = 'dataFats-oils.txt'

"""
# find number of features: 150
data = 'dataDescriptions.txt'
with open (data, 'r') as data:
	for row in data:
		string = row.split('^')
print ('The number of features is', len(string) - 1)
"""

"""
# find number of data: 1496
num_of_data = 0
with open (file1, 'r') as file1:
	for row in file1:
		num_of_data += 1
with open (file2, 'r') as file2:
	for row in file2:
		num_of_data += 1
with open (file3, 'r') as file3:
	for row in file3:
		num_of_data += 1
with open (file4, 'r') as file4:
	for row in file4:
		num_of_data += 1
		print (row)
print ('The number of data is', num_of_data)
"""
feature = np.zeros((1496, 150))
truth = np.zeros((1496))
predict = np.zeros((1496))
name = np.empty((1496), dtype = object)
min_j = np.zeros((150))
max_j = np.zeros((150))
data_index = 0

with open (file1, 'r') as file1:
	for row in file1:
		string = row.split('^')
		truth[data_index] = 0
		name[data_index] = string[0]
		for i in range (1, 151):
			feature[data_index][i-1] = string[i]
			if (data_index == 0):
				min_j[i-1] = string[i]
				max_j[i-1] = string[i]
			else:
				if (float(string[i]) < min_j[i-1]):
					min_j[i-1] = string[i]
				if (float(string[i]) > max_j[i-1]):
					max_j[i-1] = string[i]
		data_index += 1
with open (file2, 'r') as file2:
	for row in file2:
		string = row.split('^')
		truth[data_index] = 1
		name[data_index] = string[0]
		for i in range (1, 151):
			feature[data_index][i-1] = string[i]
			if (float(string[i]) < min_j[i-1]):
				min_j[i-1] = string[i]
			if (float(string[i]) > max_j[i-1]):
				max_j[i-1] = string[i]			
		data_index += 1
with open (file3, 'r') as file3:
	for row in file3:
		string = row.split('^')
		truth[data_index] = 2
		name[data_index] = string[0]
		for i in range (1, 151):
			feature[data_index][i-1] = string[i]
			if (float(string[i]) < min_j[i-1]):
				min_j[i-1] = string[i]
			if (float(string[i]) > max_j[i-1]):
				max_j[i-1] = string[i]
		data_index += 1
with open (file4, 'r') as file4:
	for row in file4:
		string = row.split('^')
		truth[data_index] = 3
		name[data_index] = string[0]
		for i in range (1, 151):
			feature[data_index][i-1] = string[i]
			if (float(string[i]) < min_j[i-1]):
				min_j[i-1] = string[i]
			if (float(string[i]) > max_j[i-1]):
				max_j[i-1] = string[i]
		data_index += 1
for i in range (1496):
	for j in range (150):
		if (max_j[j] != 0):
			feature[i][j] = (feature[i][j] - min_j[j]) / (max_j[j] - min_j[j])

# 2. 
# K-means http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

clf = KMeans(n_clusters=4)

# 3.
# Compute Rand Index
truth_random = np.random.permutation(truth)
rand_index_random = randIndex(truth, truth_random)
print ('Rand Index for a random permutation of labels is', rand_index_random)

predict = clf.fit_predict(feature)
rand_index_kmean = randIndex(truth, predict)
print ('Rand Index for K-mean labels is', rand_index_kmean) 

# 4.
# Examining K-mean objective

objective = []
rand_index_list = []

for i in range (20):
	clf = KMeans(n_clusters = 4, n_init = 1, init = 'random')
	clf.fit(feature)
	if (clf.inertia_ not in objective):
		objective.append(clf.inertia_)
		rand_index_list.append(randIndex(truth, clf.predict(feature)))

print ('The distinctive values of the objective functions are:', objective)
print ('The minimum objective function value is:', min(objective))
print ('Objective function  Rand Index')
for i in range (len(objective)):
	print (objective[i], '     ', rand_index_list[i])

"""
The distinctive values of the objective functions are: [789.38823917864681, 660.02881146305879, 805.45680053657293, 805.38798314407813, 804.12472222302642]
The minimum objective function is: 660.028811463
Objective function  Rand Index
789.388239179       0.7999633358968398
660.028811463       0.8355445066442508
805.456800537       0.7758294135531987
805.387983144       0.7772807754904941
804.124722223       0.8026612773415842
"""

# 5. 
# Dendogram plot
# Dendogram - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Linkage - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.linkage.html

short_feature = np.zeros((120, 150))
short_truth = np.zeros((120))
index_short = 0

rand_list = random.sample(range(0, 182), 30)
for i in rand_list:
	short_feature[index_short] = feature[i]
	short_truth[index_short] = truth[i]
	index_short += 1
rand_list = random.sample(range(182, 449), 30)
for i in rand_list:
	short_feature[index_short] = feature[i]
	short_truth[index_short] = truth[i]
	index_short += 1
rand_list = random.sample(range(449, 1276), 30)
for i in rand_list:
	short_feature[index_short] = feature[i]
	short_truth[index_short] = truth[i]
	index_short += 1
rand_list = random.sample(range(1277, 1496), 30)
for i in rand_list:
	short_feature[index_short] = feature[i]
	short_truth[index_short] = truth[i]
	index_short += 1

plt.figure()
hClsMat = sch.linkage(short_feature, method='complete')
sch.dendrogram(hClsMat, labels = short_truth, leaf_rotation = 45)
plt.show()

# 6. 
# Hierarchical clustering
# SciPy's Cluster - http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster

hClsMat2 = sch.linkage(feature, method='complete')
resultingClusters = sch.fcluster(hClsMat2, t= 3.8, criterion = 'distance')
print ('The Rand Index for agglomerative clustering is', randIndex(truth, resultingClusters)) # 0.5306

hClsMat2 = sch.linkage(feature, method='complete')
resultingClusters = sch.fcluster(hClsMat2, t = 2, criterion = 'distance')
print ('The Rand Index for agglomerative clustering is', randIndex(truth, resultingClusters))
print ('The number of clusters is', max(resultingClusters)) #36

# 7. 
# K-means for Sub-cluster 

for k in 5, 10, 25, 50, 75:
	print ('For k =', k)
	clf = KMeans(n_clusters = k)
	predict_sub = list(clf.fit_predict(feature[0:182]))
	largest_cluster_number = max(set(predict_sub), key = predict_sub.count)
	largest_cluster_size = predict_sub.count(largest_cluster_number)
	print ('The largest cluster size is', largest_cluster_size)
	if (largest_cluster_size < 10):
		print ('The food items in this cluster are:')
		for i in range(182):
			if (predict_sub[i] == largest_cluster_number):
				print (name[i])
	else:
		random_list = []
		print ('10 randomly sampled food items from this cluster are:')
		for i in range (182):
			if (predict_sub[i] == largest_cluster_number):
				random_list.append(name[i])
		random_list_index = random.sample(range(len(random_list)), 10)
		for j in random_list_index:
			print (random_list[j])
	print ('\n')

"""
For k = 5
The largest cluster size is 57
10 randomly sampled food items from this cluster are:
HOMINY,CANNED,YELLOW
WHEAT,SPROUTED
MILLET,COOKED
PASTA,HOMEMADE,MADE W/EGG,CKD
RICE,WHITE,MEDIUM-GRAIN,CKD
QUINOA,CKD
PASTA,CORN,COOKED
RICE,WHITE,LONG-GRAIN,REG,CKD,UNENR,WO/SALT
SPAGHETTI,CKD,ENR,W/ SALT
MACARONI,PROTEIN-FORTIFIED,CKD,ENR,(N X 6.25)


For k = 10
The largest cluster size is 57
10 randomly sampled food items from this cluster are:
RICE,WHITE,SHORT-GRAIN,CKD
NOODLES,EGG,CKD,ENR
WILD RICE,COOKED
RICE,WHITE,LONG-GRAIN,REG,CKD,ENR
SPAGHETTI,CKD,UNENR,W/ SALT
BULGUR,COOKED
AMARANTH GRAIN,CKD
PASTA,CORN,COOKED
NOODLES,EGG,CKD,ENR,W/ SALT
QUINOA,CKD


For k = 25
The largest cluster size is 30
10 randomly sampled food items from this cluster are:
MILLET,COOKED
RICE,BROWN,LONG-GRAIN,CKD
RICE,WHITE,LONG-GRAIN,PARBLD,ENR,CKD
HOMINY,CANNED,WHITE
RICE,WHITE,LONG-GRAIN,PRECKD OR INST,ENR,PREP
RICE,WHITE,SHORT-GRAIN,CKD,UNENR
RICE,BROWN,MEDIUM-GRAIN,CKD
RICE,WHITE,LONG-GRAIN,REG,CKD,ENR
RICE,WHITE,LONG-GRAIN,REG,CKD,UNENR,WO/SALT
RICE NOODLES,CKD


For k = 50
The largest cluster size is 24
10 randomly sampled food items from this cluster are:
RICE,WHITE,MEDIUM-GRAIN,CKD,UNENR
MILLET,COOKED
BARLEY,PEARLED,COOKED
TEFF,CKD
PASTA,CORN,COOKED
RICE NOODLES,CKD
RICE,WHITE,SHORT-GRAIN,CKD
RICE,WHITE,SHORT-GRAIN,CKD,UNENR
RICE,WHITE,GLUTINOUS,CKD
OAT BRAN,COOKED


For k = 75
The largest cluster size is 15
10 randomly sampled food items from this cluster are:
RICE,WHITE,LONG-GRAIN,PARBLD,UNENR,CKD
RICE,WHITE,LONG-GRAIN,REG,CKD,ENR,W/SALT
RICE,WHITE,LONG-GRAIN,REG,CKD,UNENR,WO/SALT
RICE,WHITE,MEDIUM-GRAIN,CKD,UNENR
RICE,BROWN,LONG-GRAIN,CKD
RICE,WHITE,LONG-GRAIN,REG,CKD,UNENR,W/SALT
RICE,WHITE,LONG-GRAIN,REG,CKD,ENR
PASTA,CORN,COOKED
RICE NOODLES,CKD
RICE,WHITE,SHORT-GRAIN,CKD

"""




