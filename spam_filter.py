#IMPORTS
import os
import pandas as pd
from scipy import stats
from whoosh.analysis import *

#SETTINGS
DIRECTORY = ""
OUTPUT = os.path.join(DIRECTORY, "output.csv")

###functions to make:

#1 text preparation - reading in negative , cleaning, stemming, lemitizing,
# Make a new column and put it in there - may be a new function
def text_preparation ():
    ###CODE HERE
    return

#2 feature extraction
def feature_extraction():
    ###CODE HERE
    return

#3 training the classifier + compute classifier accuracy on train/test
def train_spam_filter(X, y):
    ####CODE HERE
    k_train_results, k_test_results, classifier = k_cross_validate(X, y)

    # calculate the train mean and the 95% confidence interval for the list of results
    train_mean = np.mean(train_results)
    train_ci_low, train_ci_high = stats.t.interval(0.95, len(train_results) - 1, loc=train_mean,scale=stats.sem(train_results))

    # calculate the test mean and the 95% confidence interval for the list of results
    test_mean = np.mean(test_results)
    test_ci_low, test_ci_high = stats.t.interval(0.95, len(test_results) - 1, loc=test_mean, scale=stats.sem(test_results))
    return train_mean, train_ci_low, train_ci_high, test_mean, test_ci_low, test_ci_high

#3b k fold cross validation
def k_cross_validate(X, y, num_tests = 10):
    train_results = []
    test_results = []

    for i in range(num_tests):
        state = random.randint(1, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
        clf = LogisticRegression(C=1.0).fit(X_train, y_train)
        y_train_predict = clf.predict(X_train)
        y_test_predict = clf.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        test_accuracy = accuracy_score(y_test, y_test_predict)
        train_results.append(train_accuracy)
        test_results.append(test_accuracy)

    return train_results, test_results, clf


