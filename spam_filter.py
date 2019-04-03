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
from nltk.tag import PerceptronTagger
import nltk as nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import json

# SETTINGS
SPAM_LABELLED = rf.filePath(rf.SPAM_LABELLED)
ORIGINAL_INPUT_DATA = rf.filePath(rf.ORIGINAL_INPUT_DATA)
OUTPUT_PIPELINE = rf.filePath(rf.OUTPUT_PIPELINE)
TOP_WORDS = rf.filePath(rf.TOP_WORDS)


overlap_corpus = []  # For overlap, yes, I know global variables are bad


tagger = PerceptronTagger()
pos_tag = tagger.tag
stop = ENGLISH_STOP_WORDS


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
    df = pd.read_csv(filename, encoding="ISO-8859-1", usecols=survey_cols)
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
    # Read All Columns
    df = pd.read_csv(filename, encoding="ISO-8859-1")
    df = df.fillna('')
    if df.empty:
        print('DataFrame is empty!')
    else:
        print('Not empty!', df.shape)
    # Make a new column and put it in there - may be a new function
    df['sf_output_raw'] = df.apply(clean_feedback, axis=1)
    global overlap_corpus
    overlap_corpus = get_overlap_corpus(df)
    df['sf_output'] = df.apply(apply_stem_overlap, axis=1)
    df['sf_output_vbnn'] = df.apply(getNounsAndVerbs, axis = 1)
    df['sf_output_vbnn_phrases'] = df.apply(getNnVbPhrases, axis = 1)
    return df


def getNounsAndVerbs(row):
    text = row['Feedback']
    vbsNns = [tag[0] for tag in pos_tag(re.findall(r'\w+', text)) if ('VB' in tag[1] or 'NN' in tag[1]) and (tag[0] not in stop)]
    return ' '.join(set(vbsNns))


def getNnVbPhrases(row):
    text = row['Feedback']
    grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}               # Chunk prepositions followed by NP
    VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
    CLAUSE: {<NP><VP>}           # Chunk NP, VP
    """
    # Create phrase tree
    chunker = nltk.RegexpParser(grammar)

    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()

    # generator, generate leaves one by one
    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP' or t.label() == 'JJ' or t.label() == 'RB'):
            yield subtree.leaves()

    # stemming, lematizing, lower case...
    def normalise(word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        word = stemmer.stem(word)
        word = lemmatizer.lemmatize(word)
        return word

    # stop-words and length control
    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword."""
        accepted = bool(2 <= len(word) <= 40
                        and word.lower() not in stop)
        return accepted

    # generator, create item once a time
    def get_terms(tree):
        for leaf in leaves(tree):
            term = [normalise(w) for w, t in leaf if acceptable_word(w)]
            # Phrase only
            if len(term) > 1:
                yield term

    phrases = [word for word in get_terms(chunker.parse(pos_tag(re.findall(r'\w+', text))))]
    return json.dumps(phrases)


def clean_feedback(row):
    tokenizer = RegexTokenizer() | LowercaseFilter() | IntraWordFilter() | StopFilter()
    stemmer = StemFilter()
    combined = row['Feedback']
    lemmList = [word.decode('utf-8').split('/')[0] for word in lemmatize(combined)]
    tokenWords = [token.text for token in tokenizer(combined)]
    stemWords = [stemmer.stemfn(word) for word in tokenWords]
    final = tokenWords + lemmList + stemWords
    return ' '.join(set(final))  # Join by space so it is easy for RegexTokenizer to manage


def get_overlap_corpus(df):
    counter = Counter()
    tokenizer = RegexTokenizer()
    for index, row in df.iterrows():
        input = row['sf_output_raw']
        tokenWords = [token.text for token in tokenizer(input)]
        counter.update([word.lower() for word in tokenWords])
    corpus_first = list(counter)
    corpus_raw = sorted(corpus_first)
    new_token_corpus = []
    for word in corpus_raw:
        # Get rid of all numbers, will not overlap those
        if any(char.isdigit() for char in word):
            print("Token is a number.")
        elif len(word) < 4:
            print("Word is small, throw out before overlapping.")
        else:
            new_token_corpus.append(word)
    beginnings = []
    for word in new_token_corpus:
        for layer_word in new_token_corpus:
            if layer_word != word:
                if layer_word[:4] == word[:4]:
                    start = word[:4]
                    if start not in beginnings:
                        beginnings.append(start)
    overlap_corpus = beginnings
    return overlap_corpus


def apply_stem_overlap(row):
    global overlap_corpus
    tokenizer = RegexTokenizer()
    overlap = []
    tokenWords = [token.text for token in tokenizer(row['sf_output_raw'])]
    for beginning in overlap_corpus:
        for token in tokenWords:
            if beginning == token[:4]:
                overlap.append(beginning)
    final = overlap + tokenWords
    return ' '.join(set(final))  # Join by space so it is easy for RegexTokenizer to manage


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
# train_classifier(rf.filePath(rf.SPAM_LABELLED))
predict(OUTPUT_PIPELINE)
print('We done.')
