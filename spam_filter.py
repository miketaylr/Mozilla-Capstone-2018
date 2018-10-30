#IMPORTS
import os
import pandas as pd
from whoosh.analysis import *
import nltk
from nltk.stem import *
from collections import Counter

#SETTINGS
DIRECTORY = ""
OUTPUT_SPAM_LABELLED = os.path.join(DIRECTORY, "outputSpamLabelled.csv")

###functions to make:

#1 text preparation - reading in negative , cleaning, stemming, lemitizing,
# Make a new column and put it in there - may be a new function
def text_preparation ():
    feedbackCleaner = RegexTokenizer() | LowercaseFilter() | IntraWordFilter() \
                      | StopFilter() | StemFilter() | WordNetLemmatizer()

    num_records = 0

    survey_cols = ["Response", "ID", "Time", "Started", "Date Submitted",
                   "Status", "Language", "Referer", "Extended", "Referer", "User",
                   "Agent", "Extended", "User Agent", "Longitude", "Latitude",
                   "Country", "City", "State / Region", "Postal",
                   "Binary Sentiment", "OS", "Positive Feedback",
                   "Negative Feedback", "Relevant Site", "compound", "neg",
                   "neu", "pos", "Sites", "Issues", "Components", "Processed Feedback",
                   "IsSpam"]
    df = pd.read_csv(OUTPUT_SPAM_LABELLED, encoding="ISO-8859-1", nrows=num_records, usecols=survey_cols)
    print("Number of records to begin with: " + num_records)

    if df.empty: #need to handle empty case later
        print('DataFrame is empty!')
    else:
        print('Not empty!', df.shape)

    df['Output from spam_filter cleaning'] = df.apply(clean_feedback, axis=1)
    return df

def clean_feedback(df):
    feedbackCleaner = RegexTokenizer() | LowercaseFilter() | IntraWordFilter() \
                      | StopFilter() | StemFilter() | WordNetLemmatizer()
    for index, row in df.iterrows():
        combined = row['Positive Feedback'] + row['Negative Feedback']
        reprocessed = [token.text for token in feedbackCleaner(combined)]
    return ' '.join(set(reprocessed))

def get_top_words(df):
    tokenizer = RegexTokenizer()
    count = Counter()
    for index, row in df.iterrows():
        wordList = tokenizer.tokenize(row['Output from spam_filter cleaning'])
        count.update(wordList)
    finalWordList = [word for (word, freq) in count.most_common(2500)]
    return finalWordList

#2 feature extraction
def feature_extraction(df):
    tokenizer = RegexTokenizer()
    binary_appearance_df = []
    featureWords = get_top_words(df)
    for index, row in df.iterrows():
        wordList = tokenizer.tokenize(row['Output from spam_filter'])
        binary_appearance_df.append([1 if word in wordList else 0 for word in featureWords])
    X = pd.DataFrame(binary_appearance_df, columns=featureWords)
    return X

def get_qrel(df):
    qrel = []
    for index, row in df.iterrows():
        qrel.append(row['IsSpam'])
    y = pd.Dataframe(qrel)
    return y

#3 training the classifier + predicting + scoring
def train_spam_filter():
    ###CODE
    return

#4 pass on the classifier, make it predict pased on input
def spam_classifier():
    ####CODE HERE
    return



