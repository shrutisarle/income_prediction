# Machine Learning (CS596) Project
# Authors: Dhaval Sharma, Dhruvil Shah, Shruti Sarle, Channing Schwaebe
"""Description: This project takes a CSV file called 'adult.csv'
file as input which contains the details of various people and their salaries.
Our goal is to predict the probability of the income being greater than $50k."""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns

# Remove warnings
import warnings

warnings.filterwarnings("ignore")

# Importing the dataset
dataset = pd.read_csv('adult.csv')

#Removes incomplete entries
for column in dataset.columns:
    dataset = dataset[dataset[column] != " ?"]
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values


##############################DATA ANALYSIS####################################
# Visualize frequency distribution of income variable
f, ax = plt.subplots(1, 2, figsize=(18, 8))
ax[0] = dataset[' income'].value_counts().plot.pie(explode=[0, 0], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Income Share')
# f, ax = plt.subplots(figsize=(6, 8))
ax[1] = sns.countplot(x=" income", data=dataset, palette="Set1")
ax[1].set_title("Frequency distribution of income variable")
plt.show()

# Distribution of age variable
f, ax = plt.subplots(figsize=(10, 8))
x = dataset['age']
ax = sns.distplot(x, bins=10, color='blue')
ax.set_title("Distribution of age variable")
plt.show()

# Detect outliers in age variable with boxplot
f, ax = plt.subplots(figsize=(10, 8))
x = dataset['age']
ax = sns.boxplot(x)
ax.set_title("Visualize outliers in age variable")
plt.show()

# Visualize income with respect to age variable
f, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(x=" income", y="age", data=dataset)
ax.set_title("Visualize income with respect to age variable")
plt.show()

# Visualize income with respect to age and sex variable
plt.figure(figsize=(8, 6))
ax = sns.catplot(x=" income", y="age", col=" sex", data=dataset, kind="box", height=8, aspect=1)
plt.show()

# Pairwise relationships in dataset with respect to age, hours-per-week and capital-gain
sns.pairplot(dataset, vars=["age", " hours-per-week", " capital-gain"])
plt.show()

# Visualize age with respect to race
plt.figure(figsize=(12, 8))
sns.boxplot(x=' race', y="age", data=dataset)
plt.title("Visualize age with respect to race")
plt.show()

# Pairwise relationships in dataset with respect to sex
sns.pairplot(dataset, hue=" sex")
plt.show()

############################DATA PREPROCESSING#################################
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_x = LabelEncoder()
encode_columns = [1, 3, 5, 6, 7, 8, 9, 13]
for i in encode_columns:
    X[:, i] = labelencoder_x.fit_transform(X[:, i])
onehotencoder = OneHotEncoder(categorical_features=encode_columns)
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# Confusion Matrix
def confusion_matrix(trueY, predY):
    labels = len(np.unique(trueY))
    conf_matr = np.zeros(shape=(labels, labels)).astype('int')
    trueY = np.transpose(trueY)
    predY = np.transpose(predY)

    for i in range(len(trueY)):
        conf_matr[trueY[i]][predY[i]] += 1

    sum_of_diag = 0
    sum_of_elem = 0
    for i in range(len(conf_matr)):
        for j in range(len(conf_matr[i])):
            if i == j:
                sum_of_diag += conf_matr[i][j]
            sum_of_elem += conf_matr[i][j]
    accuracy = sum_of_diag / sum_of_elem

    precision = []
    for label in range(labels):
        column = conf_matr[:, label]
        precision.append(conf_matr[label, label] / column.sum())

    recall = []
    for label in range(labels):
        row = conf_matr[label, :]
        recall.append(conf_matr[label, label] / row.sum())

    return conf_matr, accuracy, precision, recall


print("Models Begin!")
###########################LOGISTIC REGRESSION#################################
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_logistic = logistic_classifier.predict(x_test)

# Making the Confusion Matrix
cm_logistic, logistic_accuracy, logistic_precision, logistic_recall = confusion_matrix(y_test, y_pred_logistic)
print("Confusion Matrix (Logistic Regression):\n", cm_logistic)

# Printing the Accuracy, Precision and Recall
print("Accuracy of Logistic Regression:", logistic_accuracy)
print("Precision of Logistic Regression:", logistic_precision)
print("Recall of Logistic Regression:", logistic_recall)
print("")

"""# Using self made model for training
def Sigmoid(x):
    g = float(1.0 / float((1.0 + math.exp(-1.0*x))))
    return g

##Prediction function
def Prediction(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i]*theta[i]
    return Sigmoid(z)


# implementation of cost functions
def Cost_Function(X,Y,theta,m):
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        est_yi = Prediction(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(est_yi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-est_yi)
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    print ('cost is ', J)
    return J


# gradient components called by Gradient_Descent()

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
    sumErrors = 0
    for i in range(m):
        xi = X[i]
        xij = xi[j]
        hi = Prediction(theta,X[i])
        error = (hi - Y[i])*xij
        sumErrors += error
    m = len(Y)
    constant = float(alpha)/float(m)

    J = constant * sumErrors
    return J

# execute gradient updates over thetas
def Gradient_Descent(X,Y,theta,m,alpha):
    new_theta = []
    for j in range(len(theta)):
        deltaF = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
        new_theta_value = theta[j] - deltaF
        new_theta.append(new_theta_value)
    return new_theta

theta = [0 for x in range(57)] #initial model parameters
alpha = 0.01 # learning rates
max_iteration = 3 # maximal iterations

trainx = np.ones((len(x_train), 57))
trainx[:, 1:57] = x_train[:,:]
m = len(x_train)# number of samples
for x in range(max_iteration):
    new_theta = Gradient_Descent(trainx,y_train,theta,m,alpha)
    theta = new_theta
    Cost_Function(trainx,y_train,theta,m)

#Predicting testdata
testx = np.ones((len(x_test), 57))
testx[:, 1:57] = x_test[:,:]

yHat = testx.dot(theta)
for i in range(len(yHat)):
    yHat[i] = Sigmoid(yHat[i])
yHat = (yHat >= 0.5).astype(int)

# Making the Confusion Matrix
cm_self_logistic, self_logistic_accuracy, self_logistic_precision, self_logistic_recall = confusion_matrix(y_test, yHat)
print("Confusion Matrix (Logistic Regression(Self Made)):\n", cm_self_logistic)

# Printing the Accuracy, Precision and Recall
print("Accuracy of Logistic Regression(Self Made):", self_logistic_accuracy)
print("Precision of Logistic Regression(Self Made):", self_logistic_precision)
print("Recall of Logistic Regression(Self Made):", self_logistic_recall)
print("")"""

###########################K-NEAREST NEIGHBORS#################################
# Fitting K-Nearest Neighbors to the Training set
from sklearn.neighbors import KNeighborsClassifier

kneighbors_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
kneighbors_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_kneighbors = kneighbors_classifier.predict(x_test)

# Making the Confusion Matrix
cm_kneighbors, kneighbors_accuracy, kneighbors_precision, kneighbors_recall = confusion_matrix(y_test,
                                                                                               y_pred_kneighbors)
print("Confusion Matrix (K-Nearest Neighbors):\n", cm_kneighbors)

# Printing the Accuracy, Precision and Recall
print("Accuracy of K-Nearest Neighbors:", kneighbors_accuracy)
print("Precision of K-Nearest Neighbors:", kneighbors_precision)
print("Recall of K-Nearest Neighbors:", kneighbors_recall)
print("")

###################################SVM#########################################
"""# Fitting SVM to the Training set
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf')
svm_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_svm = svm_classifier.predict(x_test)

# Making the Confusion Matrix
cm_svm, svm_accuracy, svm_precision, svm_recall = confusion_matrix(y_test, y_pred_svm)
print("Confusion Matrix (SVM):\n", cm_svm)

# Printing the Accuracy, Precision and Recall
print("Accuracy of SVM:", svm_accuracy)
print("Precision of SVM:", svm_precision)
print("Recall of SVM:", svm_recall)
print("")"""

#########################DECISION TREE CLASSIFICATION##########################
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier

decision_tree_classifier = DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_decision_tree = decision_tree_classifier.predict(x_test)

# Making the Confusion Matrix
cm_decision_tree, decision_tree_accuracy, decision_tree_precision, decision_tree_recall = confusion_matrix(y_test,
                                                                                                           y_pred_decision_tree)
print("Confusion Matrix (Decision Tree):\n", cm_decision_tree)

# Printing the Accuracy, Precision and Recall
print("Accuracy of Decision Tree:", decision_tree_accuracy)
print("Precision of Decision Tree:", decision_tree_precision)
print("Recall of Decision Tree:", decision_tree_recall)
print("")

########################RANDOM FOREST CLASSIFIER###############################
# Fitting Random Forest Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier

random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
random_forest_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_random_forest = random_forest_classifier.predict(x_test)

# Making the Confusion Matrix
cm_random_forest, random_forest_accuracy, random_forest_precision, random_forest_recall = confusion_matrix(y_test,
                                                                                                           y_pred_random_forest)
print("Confusion Matrix (Random Forest Classifier):\n", cm_random_forest)

# Printing the Accuracy, Precision and Recall
print("Accuracy of Random Forest Classifier:", random_forest_accuracy)
print("Precision of Random Forest Classifier:", random_forest_precision)
print("Recall of Random Forest Classifier:", random_forest_recall)
print("")

########################NAIVE BAYES###############################
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred_naive_bayes = naive_bayes_classifier.predict(x_test)

# Making the Confusion Matrix
cm_naive_bayes, naive_bayes_accuracy, naive_bayes_precision, naive_bayes_recall = confusion_matrix(y_test,
                                                                                                   y_pred_naive_bayes)
print("Confusion Matrix (Naive Bayes Classifier):\n", cm_naive_bayes)

# Printing the Accuracy, Precision and Recall
print("Accuracy of Naive Bayes Classifier:", naive_bayes_accuracy)
print("Precision of Naive Bayes Classifier:", naive_bayes_precision)
print("Recall of Naive Bayes Classifier:", naive_bayes_recall)
print("")

##########################ARTIFICIAL NEURAL NETWORKS###########################
print("Artificial Neural Networks Starts!")
# Initialising the ANN
ann_classifier = Sequential()

# Adding the input layer and the first hidden layer
ann_classifier.add(Dense(output_dim=200, init='uniform', activation='relu', input_dim=104))

# Adding the second hidden layer
ann_classifier.add(Dense(output_dim=200, init='uniform', activation='relu'))

# Adding the output layer
ann_classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
ann_classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
ann_classifier.fit(x_train, y_train, batch_size=50, nb_epoch=20)

# Predicting the Test set results
y_pred_ann = ann_classifier.predict(x_test)
y_pred_ann = (y_pred_ann > 0.5)
y_pred_ann = np.transpose(y_pred_ann).astype('int32')

# Making the Confusion Matrix
cm_ann, ann_accuracy, ann_precision, ann_recall = confusion_matrix(y_test, y_pred_ann)
print("Confusion Matrix (ANN):\n", cm_ann)

# Printing the Accuracy, Precision and Recall
print("Accuracy of ANN:", ann_accuracy)
print("Precision of ANN:", ann_precision)
print("Recall of ANN:", ann_recall)

from categories import categories

categories()