# IMPORTS
import os as os
import pandas as pd
from scipy import stats
from whoosh.analysis import *
import numpy as np
import random as random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from collections import Counter
import pickle as pickle
import referenceFiles as rf
from pattern.en import *
from gensim.utils import lemmatize

# SETTINGS
SPAM_LABELLED = rf.filePath(rf.SPAM_LABELLED)
ORIGINAL_INPUT_DATA = rf.filePath(rf.ORIGINAL_INPUT_DATA)
OUTPUT_PIPELINE = rf.filePath(rf.OUTPUT_PIPELINE)
TOP_WORDS = rf.filePath(rf.TOP_WORDS)


# 1 Text Preparation
def text_preparation(filename):
    num_records = 5000
    survey_cols = ["Response ID", "Time Started", "Date Submitted",
                   "Status", "Language", "Referer", "Extended Referer", "User Agent",
                   "Extended User Agent", "Longitude", "Latitude",
                   "Country", "City", "State/Region", "Postal",
                   "Binary Sentiment", "OS", "Positive Feedback",
                   "Negative Feedback", "Relevant Site", "compound", "neg",
                   "neu", "pos", "Sites", "Issues", "Components", "Processed Feedback",
                   "IsSpam"]
    df = pd.read_csv(filename, encoding="ISO-8859-1", nrows=num_records, usecols=survey_cols)
    df = df.fillna('')
    if df.empty:
        print('DataFrame is empty!')
    else:
        print('Not empty!', df.shape)
    # Make a new column and put it in there - may be a new function
    df['sf_output'] = df.apply(clean_feedback, axis=1)
    return df


def text_preparation_unlabelled(filename):
    num_records = 5000
    survey_cols = ["Response ID", "Time Started", "Date Submitted",
                   "Status", "Language", "Referer", "Extended Referer", "User Agent",
                   "Extended User Agent", "Longitude", "Latitude",
                   "Country", "City", "State/Region", "Postal",
                   "Binary Sentiment", "OS", "Positive Feedback",
                   "Negative Feedback", "Relevant Site", "compound", "neg",
                   "neu", "pos", "Sites", "Issues", "Components", "Processed Feedback"]
    df = pd.read_csv(filename, encoding="ISO-8859-1", nrows=num_records, usecols=survey_cols)
    df = df.fillna('')
    if df.empty:
        print('DataFrame is empty!')
    else:
        print('Not empty!', df.shape)
    # Make a new column and put it in there - may be a new function
    df['sf_output'] = df.apply(clean_feedback, axis=1)
    # token_corpus = get_token_corpus(df)
    # df['overlap'] = df.apply(apply_stem_overlap, token_corpus, axis=1)
    return df


def clean_feedback(row):
    tokenizer = RegexTokenizer() | LowercaseFilter() | IntraWordFilter() | StopFilter()
    stemmer = StemFilter()
    combined = row['Positive Feedback'] + row['Negative Feedback']  # Feedback will either be positive or negative
                                                                    # can filter for negative only later on
    lemmList = [word.decode('utf-8').split('/')[0] for word in lemmatize(combined)]
    tokenWords = [token.text for token in tokenizer(combined)]
    stemWords = [stemmer.stemfn(word) for word in tokenWords]
    final = tokenWords + lemmList + stemWords
    return ' '.join(set(final))  # Join by space so it is easy for RegexTokenizer to manage


def get_token_corpus(df):
    counter = Counter()
    for index, row in df.iterrows():
        input = row['sf_output']
        counter.update([word.lower() for word in re.findall(r'\\w+', input)])
    corpus = list(counter)
    return corpus


def apply_stem_overlap(row, token_corpus):
    # TODO
    # https://stackoverflow.com/questions/49088978/how-to-create-corpus-from-pandas-data-frame-to-operate-with-nltk
    final = []
    return ' '.join(set(final))


def get_top_words(df):
    tokenizer = RegexTokenizer()
    count = Counter()
    for index, row in df.iterrows():
        wordList = [token.text for token in tokenizer(row['sf_output'])]
        count.update(wordList)
    totalNumWords = sum(count.values())
    numUniqueWords = len(count)
    finalWordList = [word for (word, freq) in count.most_common(round(totalNumWords/10))]
    # TODO:
    #       Discuss whether it is a good idea to take top words (might it actually
    #       be better to take least common?) Just an idea...
    with open(TOP_WORDS, 'wb') as fp:
        pickle.dump(finalWordList, fp)
    return finalWordList


# 2 Feature Extraction
def feature_extraction(df, get_new_words=True):
    tokenizer = RegexTokenizer()
    binary_appearance_df = []
    if (get_new_words):
        featureWords = get_top_words(df)
    else:
        with open(TOP_WORDS, 'rb') as fp:
            featureWords = pickle.load(fp)
    for index, row in df.iterrows():
        wordList = [token.text for token in tokenizer(row['sf_output'])]
        binary_appearance_df.append([1 if word in wordList else 0 for word in featureWords])
    X = pd.DataFrame(binary_appearance_df, columns=featureWords)
    # print(X)
    return X


def get_qrel(df):
    qrel = []
    for index, row in df.iterrows():
        qrel.append(row['IsSpam'])
    # y = pd.DataFrame(qrel)
    return qrel


# 3 Train the Classifier
def train_spam_filter(X, y, num_tests = 10):
    train_results, test_results, train_recall, test_recall, clf = k_cross_validate(X, y, num_tests)
    # print recall scores
    print_recall_scores(train_recall, test_recall)
    # calculate the train mean and the 95% confidence interval for the list of results
    train_mean = np.mean(train_results)
    train_ci_low, train_ci_high = stats.t.interval(0.95, len(train_results) - 1, loc=train_mean,scale=stats.sem(train_results))
    # calculate the test mean and the 95% confidence interval for the list of results
    test_mean = np.mean(test_results)
    test_ci_low, test_ci_high = stats.t.interval(0.95, len(test_results) - 1, loc=test_mean, scale=stats.sem(test_results))
    return clf, train_mean, train_ci_low, train_ci_high, test_mean, test_ci_low, test_ci_high


def k_cross_validate(X, y, num_tests):
    train_results = []
    test_results = []
    train_recall_all = []
    test_recall_all = []
    for i in range(num_tests):
        state = random.randint(1, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
        clf = LogisticRegression(C=1.0).fit(X_train, y_train)
        y_train_predict = clf.predict(X_train)
        y_test_predict = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_predict)
        train_accuracy = accuracy_score(y_train, y_train_predict)
        test_recall = recall_score(y_test, y_test_predict)
        train_recall = recall_score(y_train, y_train_predict)

        train_results.append(train_accuracy)
        test_results.append(test_accuracy)
        train_recall_all.append(train_recall)
        test_recall_all.append(test_recall)
    return train_results, test_results, train_recall_all, test_recall_all, clf


def print_recall_scores(train_recall, test_recall):
    print("Recall Scores:")
    # print("Train: ", train_recall)
    print("Train avg: ", sum(train_recall) / float(len(train_recall)))
    # print("Test: ", test_recall)
    print("Test avg: ", sum(test_recall) / float(len(test_recall)))
    return


def save_classifier_model(clf):
    filename = "spamClassifier.sav"
    pickle.dump(clf, open(filename, 'wb'))


def load_classifier(filename):
    return pickle.load(open(filename, 'rb'))


def score_new_data(clf, X):
    return clf.predict(X)


def get_nonspam_indices(y):
    npArr = np.array(y)
    x = np.where(npArr == 0)[0]
    return x


def remove_spam(df, nsi):
    new_df = df.iloc[nsi, :]
    return new_df


def performance_metrics_spam_removal():
    return

print('We starting.')

def train_classifier(filePath):
    # Get the model and check its accuracy
    # Train Classifier on labelled data
    # NOTE: run this the first time if you don't have the classifier built
    df = text_preparation(filePath)
    X = feature_extraction(df)
    y = get_qrel(df)
    clf, train_mean, train_ci_low, train_ci_high, test_mean, test_ci_low, test_ci_high = train_spam_filter(X, y)
    save_classifier_model(clf)
    print(train_mean, train_ci_low, train_ci_high, test_mean, test_ci_low, test_ci_high)
    return


def predict(INPUT):
    # Predict and remove spam in new csv
    loaded_clf = load_classifier("spamClassifier.sav")
    new_df = text_preparation_unlabelled(INPUT)
    X = feature_extraction(new_df, False)
    y = score_new_data(loaded_clf, X)
    nsi = get_nonspam_indices(y).tolist()
    remove_spam(new_df, nsi).to_csv(rf.filePath(rf.OUTPUT_SPAM_REMOVAL))
    return


print('We starting.')
train_classifier(rf.filePath(rf.SPAM_LABELLED))
predict(OUTPUT_PIPELINE)
print('We done.')
