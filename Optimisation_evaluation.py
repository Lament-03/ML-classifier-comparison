# Description: Performing optimisation and evaluation of classifiers using k-fold cross validation and confusion matrices
# Created on: 06/03/2025

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

numbers = datasets.load_digits() # Loading the digits dataset from sklearn

X = numbers.data 
y = numbers.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 47) # Splitting the dataset into training and testing sets

# Perfoming a k-fold cross validaton on each classifier
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X_train, y_train)
scores = cross_val_score(KNN, X, y, cv=10, scoring="accuracy")
print("KNN Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Making a confusion matrix for the KNN classifier
y_pred = KNN.predict(X_test)
ConfusionMatrix = metrics.confusion_matrix(y_test, y_pred)
DisplayMatrix = metrics.ConfusionMatrixDisplay(ConfusionMatrix)
DisplayMatrix.plot()
plt.title("K Nearest Neighbour Confusion Matrix")
plt.savefig("knn_confusion_matrix.png")
plt.show()

# Random forest classifier with a max sample size of 10 and a random state of 47
RandomForest = RandomForestClassifier(max_samples=10, random_state=47)
RandomForest.fit(X_train, y_train)
scores = cross_val_score(RandomForest, X, y, cv=10, scoring="accuracy")
print("Random Forest Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = RandomForest.predict(X_test)
ConfusionMatrix = metrics.confusion_matrix(y_test, y_pred)
DisplayMatrix = metrics.ConfusionMatrixDisplay(ConfusionMatrix)
DisplayMatrix.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("random_forest_confusion_matrix.png")
plt.show()

DecisionTree = DecisionTreeClassifier(criterion = "entropy", max_depth = 10, max_features = 10, random_state = 47)
DecisionTree.fit(X_train, y_train)
scores = cross_val_score(DecisionTree, X, y, cv=10, scoring="accuracy")
print("Decision Tree Average Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

y_pred = DecisionTree.predict(X_test)
ConfusionMatrix = metrics.confusion_matrix(y_test, y_pred)
DisplayMatrix = metrics.ConfusionMatrixDisplay(ConfusionMatrix)
DisplayMatrix.plot()
plt.title("Decision Tree Confusion Matrix")
plt.savefig("decision_tree_confusion_matrix.png")
plt.show()
max_depths = 0

# Empty arrays to store 10 fold accuracy values
KNN_Averages = []
RandomForest_Averages = []
DecisionTree_Averages = []


# Performing 10 fold cross validation for each classifier attribute value changed
for i in range(5, 11):
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train, y_train)
    scores = cross_val_score(KNN, X, y, cv=10, scoring="accuracy")
    print("KNN Average Accuracy for " + str(i) + ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    KNN_Averages.append(scores.mean())
    
    RandomForest = RandomForestClassifier(max_samples=i, random_state=47)
    RandomForest.fit(X_train, y_train)
    scores = cross_val_score(RandomForest, X, y, cv=10, scoring="accuracy")
    print("Random Forest Average Accuracy for " + str(i) + ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    RandomForest_Averages.append(scores.mean())
    
    DecisionTree = DecisionTreeClassifier(criterion = "entropy", max_depth = i, max_features = 10, random_state = 47)
    DecisionTree.fit(X_train, y_train)
    scores = cross_val_score(DecisionTree, X, y, cv=10, scoring="accuracy")
    print("Decision Tree Average Accuracy for " + str(i) + ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    DecisionTree_Averages.append(scores.mean())
    
y1 = np.array(KNN_Averages)
y2 = np.array(RandomForest_Averages)
y3 = np.array(DecisionTree_Averages)

# x_axis = [5, 6, 7, 8, 9, 10]
# plt.xlabel("Parameter values")
# plt.ylabel("Average accuracy")
# plt.plot(x_axis, KNN_Averages, label = "K Nearest Neighbour")
# plt.plot(x_axis, RandomForest_Averages, label = "Random Forest")
# plt.plot(x_axis, DecisionTree_Averages, label = "Decision Tree")
# plt.legend(loc = "lower right")
# plt.show()

plt.figure(figsize=(9,9))

bar_width = 0.25

br1 = np.arange(len(KNN_Averages))
br2 = [x + bar_width for x in br1] 
br3 = [x + bar_width for x in br2] 
 
# Make the plot
plt.title("Average accuracy of classifiers with the same set parameter")
plt.bar(br1, KNN_Averages, color ='r', width = bar_width, edgecolor ='grey', label ='K Nearest Neighbour') 
plt.bar(br2, RandomForest_Averages, color ='g', width = bar_width, edgecolor ='grey', label ='Random Forest')
plt.bar(br3, DecisionTree_Averages, color ='blue', width = bar_width, edgecolor ='grey', label ='Decision Tree')
 
# Adding the labels
plt.xlabel('Parameter Values', fontweight ='bold', fontsize = 15) 
plt.ylabel('Average accuracy', fontweight ='bold', fontsize = 15) 
plt.legend(loc="upper right")
plt.xticks([r + bar_width for r in range(len(KNN_Averages))], ["5", "6", "7", "8", "9", "10"])


plt.show() #KNN received the highest average accuracy of 97%