# Income Classification
## A Case Study on The Efficacy of Machine Learning Models in Predicting Income
 
This project evaluates the effectiveness of multiple machine learning models in predicting income based on data extracted from the 1994 US Census database by Ronny Kohavi and Barry Becker and hosted by UCI Machine Learning.1 The models tested include Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests, Naïve Bayes Classifiers, and Artificial Neural Networks. Through our analysis of these models on this problem instance, we found that a Random Forest generated the most accurate predictions. Additionally, the data was separated by category to determine the relative strength of each category in predicting income.  In doing so, we found that investment performance alone (capital gains and losses) was the most accurate predictor of income.

This Machine Learning project uses data from the 1994 US Census to predict income based on a set of 14 parameters. In this dataset, income was binned into two classes, less than $50,000 and greater than or equal to $50,000. This binary classification problem was evaluated using six popular Machine Learning algorithms: Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests, Naïve Bayes Classifiers, and Artificial Neural Networks. Of these models, a Random Forest was found to be the most accurate. This is largely due to its ability to accurately predict the minority class (income greater than or equal to $50,000). 

#### Task Description
Our task was to build and train several different models on the dataset. The models were then given a test set and tasked with predicting the income for individuals in the test set. The results were analyzed to determine the best model for this binary classification problem. Once we were able to determine the most accurate model, we set out to determine the most accurate subset of data. To do this, we split the parameters into four bins and ran the model on each subset. 

#### Dataset Description
In this dataset, income is defined as “income received on a regular basis (exclusive of certain money receipts such as capital gains) before payments for personal income taxes, social security, union dues, Medicare deductions, etc.” 2 Income was separated into two classes:

1.)	Less than $50,000
2.)	Greater than or equal to $50,000
 
The original dataset contained 14 features, which were used to predict the aforementioned label, income. The features are as follows:

1.)	Age - Continuous
2.)	Workclass – Categorical 
3.)	Fnlwgt - Continuous
4.)	Education - Categorical
5.)	Education-num - Continuous
6.)	Marital-Status - Categorical
7.)	Occupation - Categorical
8.)	Relationship - Categorical
9.)	Race - Categorical
10.)	Sex - Categorical
11.)	Capital-Gain - Continuous
12.)	Capital-Loss - Continuous
13.)	Hours-per-Week - Continuous
14.)	Native-Country – Categorical

The secondary evaluations of subsets of the original data in predicting income consisted of four categories:

1.)	Education – (Education-num)
2.)	Occupation – (Workclass, Occupation, Hours-per-Week)
3.)	Demographic – (Age, Marital-Status, Relationship, Race, Sex, Native-Country)
4.)	Investments – (Capital-Gain, Capital-Loss)

#### Conclusion
Through our analysis of the income classification dataset, we found that a Random Forest was the most accurate model for predicting income. However, the dataset was highly unbalanced, which caused every model to be undertrained for the minority class. Due to this fact, we cannot conclusively state that a Random Forest is the optimal model for this type of classification problem. It is possible that with a larger, more balanced dataset, another model would have been more accurate. But for this dataset, the optimal model was a Random Forest.  
The findings in our subset evaluations provide an intriguing hypothesis for future research. Much more data would need to be gathered and a more even distribution of samples would be beneficial for training the models. A dataset with more granularity in income would also be an improvement as the current classification groups a wide array of socio-economic classes into just two groups.

