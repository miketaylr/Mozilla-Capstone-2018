# Pipeline package for extracting raw data and doing cleaning
import pandas as pd
import numpy as np
import re
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE
from whoosh.analysis import *
import referenceFiles as rf
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import tldextract
import nltk
nltk.downloader.download('vader_lexicon')

#Progress bar
from tqdm import tqdm

# FOR NOW just lower the terms in the dicts. Need to see how stemming and more can play into this
WTI = {k: re.compile('|'.join(v).lower()) for k, v in WORDS_TO_ISSUE.items()}
WTC = {k: re.compile('|'.join(v).lower()) for k, v in WORDS_TO_COMPONENT.items()}

# Read in the brands-keywords mapping csv
brands = pd.read_csv(rf.filePath(rf.BRAND_KEYWORDS))
# Convert to dictionary
WTB = brands.set_index('Brand').T.to_dict('list')
WTB = {k: v[0] for k, v in WTB.items()}
WTB = {k: re.compile('|'.join(v.split(',')).lower()) for k, v in WTB.items()}
# Clean up the raw dictionaries a bit more eventually, fix typos etc.


# Need to make num_records optional arg, s.t. if user wants all the records they don't specify a number
def run_pipeline(top_sites_location, raw_data_location, num_records=-1):
    print(
        "Hi there, I am your pipeline slave. Your wish is my command.\nI am getting the data and cleaning it for you :)\n")

    # for columns A and B in the top 100, get strings in cells, comma, split by comma,
    # then save and check if data contains these values
    print("Loading top sites from %s " % top_sites_location)
    sites = pd.read_csv(top_sites_location, usecols=['Domains', 'Brand'])
    siteList = list(sites.values.flatten())

    # remove commas ('salesforce.com, force.com')
    for site in siteList:
        if ',' in site:
            siteList += site.split(',')

    siteList = [site.strip('.*') for site in list(filter(lambda site: ',' not in site, siteList))]
    siteList = '|'.join(siteList)
    siteListRegex = re.compile(siteList + '|https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
    # Filter we will initially use these filters
    stmLwrFilter = RegexTokenizer() | StemFilter() | LowercaseFilter()
    # lwrFilter = LowercaseFilter()

    # read in raw survey data from CSV files. Only want certain columns
    survey_cols = ["Response ID", "Date Submitted", "Language", "Country",
                    "How does Firefox make you feel?",
                   "To help us understand your input, we need more information. Please describe what you like. The content of your feedback will be public, so please be sure not to include personal information such as email address, passwords or phone number.",
                   "To help us understand your input, we need more information. Please describe your problem below and be as specific as you can. The content of your feedback will be public, so please be sure not to include personal information such as email address, passwords or phone number.",
                   "If your feedback is related to a website, you can include it here:"]
    if (num_records < 0):  # return all records
        df = pd.read_csv(raw_data_location, encoding="ISO-8859-1", usecols=survey_cols)
    else:
        df = pd.read_csv(raw_data_location, encoding="ISO-8859-1", usecols=survey_cols)
        df = df.loc[~df['Country'].isnull()]
        df = df.tail(num_records)

    print(df['Country'])

    # some data cleaning and selection
    print("Loading %d feedback records from %s " % (num_records, raw_data_location))
    # rename some long column names
    df.rename(columns={survey_cols[4]: 'Binary Sentiment', survey_cols[5]: 'Positive Feedback',
                       survey_cols[6]: 'Negative Feedback', survey_cols[7]: 'Relevant Site'}, inplace=True)
    df = df.fillna('');  # repalce NaNs with blanks
    df = df.loc[df['Language'] == 'English']  # Only want english rows
    df = df.loc[~df['Negative Feedback'].str.contains('[À-ÿ]')]  # Only want rows without accented characters

    print("After filtering for English, only %d records remain" % len(df.index))
    # really basic spam filtering.  We can look at these separately and make spam more robust
    df = df.loc[(df['Positive Feedback'] + df[
        'Negative Feedback']).str.len() > 20]  # If the length of the feedback is over 1k characters OR is less than 20 characters then it is spam.
    print("After basic spam filtering only %d records remain" % len(df.index))
    # Convert to df friendly date-times
    df["Date Submitted"] = pd.to_datetime(df["Date Submitted"])

    #filtered_out.to_csv(rf.filePath("data/filtered_out.csv"), encoding='ISO-8859-1')
    if df.empty:  # need to handle empty case later
        print('DataFrame is empty!')
    else:
        print('Not empty!', df.shape)

    # crude way of looking for mentioned site using the top 100 list. Need to add the regex to pick up wildcard sites
    def mentioned_site(row):
        combined = row['Feedback'].lower() + ' ' + row['Relevant Site'].lower()
        #sites = [site.lower() for site in siteList if site.lower() in combined]
        urls = re.findall(siteListRegex, combined) #NEED TO IMPROVE REGEX TO PICK UP MORE SITES
        return ','.join(list(set(urls)))

    # crude way of looking for mentioned site using the top 100 list. Need to add the regex to pick up wildcard sites
    def mentioned_brand(row):
        combined = row['Feedback'].lower() + ' ' + row['Relevant Site'].lower()
        brands = [k for k, v in WTB.items() if v.search(combined)]
        #look at sites column for any extracted urls if we dont catch any brands on the first pass
        if (len(brands) == 0):
            sites = row['Sites'].split(',')
            brands = [tldextract.extract(v).domain for v in sites]

        return ','.join(list(set(brands)))

        # Find a mentioned issue based on our issues dictionary

    def mentioned_component(row):
        combined = row['Feedback'].lower()
        components = [k for k, v in WORDS_TO_COMPONENT.items() if any(map(lambda term: term in combined, v))]

        return ','.join(set(components))

    def apply_and_concat(dataframe, field, func, column_names):
        return pd.concat((
            dataframe,
            dataframe[field].apply(
                lambda cell: pd.Series(func(cell), index=column_names))), axis=1)
        # example usage: apply_and_concat(df, 'A', func, ['x^2', 'x^3'])

    # basic sentiment analysis
    # Use vader to evaluated sentiment of reviews
    def evalSentences(sia, sentence): #only one sentence
        # Instantiate an instance to access SentimentIntensityAnalyzer class
        sid = sia
        ss = sid.polarity_scores(sentence) # run sentiment on the sentence
        return [ss['compound']] # return the value of the compound sentiment


    def derive_columns(data_frame):  # based on cols from data after cleaning

        data_frame['Feedback'] = data_frame['Positive Feedback'].map(str) + data_frame['Negative Feedback'].map(str)
        data_frame['Sites'] = data_frame.apply(mentioned_site, axis=1)
        data_frame['Brands'] = data_frame.apply(mentioned_brand, axis=1)

        data_frame = data_frame.merge(
            df['Feedback'].apply(lambda s: pd.Series({'Issues': [k for k, v in WTI.items() if v.search(s)]})),
            left_index=True, right_index=True)

        data_frame = data_frame.merge(
            df['Feedback'].apply(lambda s: pd.Series({'Components': [k for k, v in WTC.items() if v.search(s)]})),
            left_index=True, right_index=True)

        sid = SIA()

        data_frame = data_frame.merge(
            df['Feedback'].apply(lambda s: pd.Series({'compound': evalSentences(sid, s)[0]})),
            left_index=True, right_index=True)

        return data_frame

    # Initialize and derive new columns
    df = derive_columns(df)

    # Delete columns we don't need anymore
    df.drop('Positive Feedback', axis=1, inplace=True)
    df.drop('Negative Feedback', axis=1, inplace=True)

    # finally output the cleaned data to a CSV
    df.to_csv(rf.filePath(rf.OUTPUT_PIPELINE), encoding='ISO-8859-1')
    print("Outputted cleaned data to output_pipeline.csv")


run_pipeline(rf.filePath(rf.SITES), rf.filePath(rf.ORIGINAL_INPUT_DATA), 20000)
