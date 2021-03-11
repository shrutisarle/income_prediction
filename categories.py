# Machine Learning (CS596) Project
# Authors: Dhaval Sharma, Dhruvil Shah, Shruti Sarle, Channing Schwaebe
'''Splits the dataset into 4 categories (education, occupation, demographics, and investments)
 to determine the relative influence of each category on income. Only uses Random Forest
because it was found to be the most effective algorithm when using the full dataset.'''

# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

def evaluate(X, y, category):
    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Fitting Random Forest Classifier to the Training set
    random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    random_forest_classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred_random_forest = random_forest_classifier.predict(x_test)

    # Making the Confusion Matrix
    #cm_random_forest, random_forest_accuracy, random_forest_precision, random_forest_recall = confusion_matrix(y_test,y_pred_random_forest)
    #print("Confusion Matrix (Random Forest Classifier):\n", cm_random_forest)

    cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)
    print("Confusion Matrix for", category)
    print(cm_random_forest)

    # Printing the Accuracy, Precision and Recall
    print("Accuracy of", category, accuracy_score(y_test, y_pred_random_forest))
    print("Precision of", category, precision_score(y_test, y_pred_random_forest, average=None))
    print("Recall of", category, recall_score(y_test, y_pred_random_forest, average=None))
    print("")

# Importing the dataset
def categories():
    dataset = pd.read_csv('adult.csv')

    for column in dataset.columns:
        dataset = dataset[dataset[column] != " ?"]

    y = pd.get_dummies(dataset.filter(items=[' income']))
    y = y.iloc[:, -1].values

    education = dataset.filter(items=[' education-num'])
    education = education.iloc[:, :].values

    occupation = pd.get_dummies(dataset.filter(items=[' workclass', ' occupation', ' hours-per-week']))
    occupation = occupation.iloc[:, :].values

    demographic = pd.get_dummies(dataset.filter(items=['age', ' marital-status', ' relationship', ' race', ' sex', ' native-country']))
    demographic = demographic.iloc[:, :].values

    investments = dataset.filter(items=[' capital-gain', ' capital-loss'])
    investments = investments.iloc[:, :].values

    evaluate(education, y, "Education")
    evaluate(occupation, y, "Occupation")
    evaluate(demographic, y, "Demographics")
    evaluate(investments, y, "Investments")

categories()



