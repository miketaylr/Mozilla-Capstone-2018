# Pipeline package for extracting raw data and doing cleaning
import pandas as pd
import re
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE
from whoosh.analysis import *
import referenceFiles as rf


# FOR NOW just lower the terms in the dicts. Need to see how stemming and more can play into this
WORDS_TO_ISSUE = {k: (map(lambda word: word.lower(), v)) for k, v in WORDS_TO_ISSUE.items()}
WORDS_TO_COMPONENT = {k:(map(lambda word: word.lower(), v)) for k, v in WORDS_TO_COMPONENT.items()}
# Clean up the raw dictionaries a bit more eventually, fix typos etc.


# Need to make num_records optional arg, s.t. if user wants all the records they don't specify a number
def run_pipeline(top_sites_location, raw_data_location, num_records):
    print(
        "Hi there, I am your pipeline slave. Your wish is my command.\nI am getting the data and cleaning it for you :)\n")

    # for columns A and B in the top 100, get strings in cells, comma, split by comma,
    # then save and check if data contains these values
    print ("Loading top sites from %s " % top_sites_location)
    sites = pd.read_csv(top_sites_location, usecols=['Domains', 'Brand'])
    # , skiprows = 50, nrows = 25
    # display(sites)
    siteList = list(sites.values.flatten())

    # remove commas ('salesforce.com, force.com')
    for site in siteList:
        if ',' in site:
            siteList += site.split(',')

    siteList = [site.strip('.*') for site in list(filter(lambda site: ',' not in site, siteList))]

    # Filter we will initially use these filters
    stmLwrFilter = RegexTokenizer() | StemFilter() | LowercaseFilter()
    # lwrFilter = LowercaseFilter()

    # read in raw survey data from CSV files. Only want certain columns
    survey_cols = ["Response ID","Time Started","Date Submitted","Status","Language","Referer","Extended Referer","User Agent","Extended User Agent","Longitude","Latitude","Country","City","State/Region","Postal","How does Firefox make you feel?","OS","To help us understand your input, we need more information. Please describe what you like. The content of your feedback will be public, so please be sure not to include personal information such as email address, passwords or phone number.","To help us understand your input, we need more information. Please describe your problem below and be as specific as you can. The content of your feedback will be public, so please be sure not to include personal information such as email address, passwords or phone number.","If your feedback is related to a website, you can include it here:"]
    df = pd.read_csv(raw_data_location, encoding ="ISO-8859-1", nrows=num_records, usecols=survey_cols)
    # some data cleaning and selection
    print("Loading %d feedback records from %s " % (num_records, raw_data_location))
    # rename some long column names
    df.rename(columns={ survey_cols[15]: 'Binary Sentiment', survey_cols[17]: 'Positive Feedback', survey_cols[18]: 'Negative Feedback', survey_cols[19]: 'Relevant Site'}, inplace=True)
    df = df.fillna(''); # repalce NaNs with blanks
    df = df.loc[df['Status'] == 'Complete'] # Only want completed surveys
    df = df.loc[df['Language'] == 'English'] # Only want english rows
    print("After filtering for English, only %d records remain" % len(df.index))
    # really basic spam filtering.  We can look at these separately and make spam more robust
    df = df.loc[(df['Positive Feedback']+df['Negative Feedback']).str.len() > 20] # If the length of the feedback is over 1k characters OR is less than 20 characters then it is spam.
    print("After basic spam filtering only %d records remain" % len(df.index))
    # Convert to df friendly date-times
    df["Date Submitted"] = pd.to_datetime(df["Date Submitted"])
    df["Time Started"] = pd.to_datetime(df["Time Started"]) # probably don't need this anymore

    if df.empty: # need to handle empty case later
        print('DataFrame is empty!')
    else:
        print('Not empty!', df.shape)

    # start with basic sentiment analysis
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

    analyzer = SIA()
    results = []

    # just append & analyze the -ve/+ve feedback for now if user gave both
    # df[['Neg', 'Neu', 'Pos', 'Compound']] = df['Text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in df.iterrows():
        pol_score = analyzer.polarity_scores(row['Positive Feedback'] + row['Negative Feedback'])
        results.append(pol_score)

    df2 = pd.DataFrame.from_records(results)
    print("df-sent size", df2.shape)
    df = df.join(df2) # this was the bug!
    print('after  sentiment', df.shape)

    def apply_nlp(row):
        combined = row['Positive Feedback'] + row['Negative Feedback']
        filtered = [token.text for token in stmLwrFilter(combined)]
        return ','.join(set(filtered) ) # turn to set temporarily to get unique values

    # crude way of looking for mentioned site using the top 100 list. Need to add the regex to pick up wildcard sites
    def mentioned_site(row):
        combined = row['Relevant Site'] + row['Positive Feedback'] + row['Negative Feedback']
        combined = combined.lower()
        sites = [site.lower() for site in siteList if site.lower() in combined]
        return ','.join(set(sites))

        # Find a mentioned issue based on our issues dictionary

    def mentioned_issue(row): # here the values in the dict are already lowered from before
        combined = row['Positive Feedback'] + row['Negative Feedback']
        issues = [k for k, v in WORDS_TO_ISSUE.items() if any(map(lambda term: term in combined.lower(), v))]
        return ','.join(set(issues))

    def mentioned_component(row):
        combined = row['Positive Feedback'] + row['Negative Feedback']
        components = [k for k, v in WORDS_TO_COMPONENT.items() if any(map(lambda term: term in combined.lower(), v))]
        return ','.join(set(components))


    # Initialize and derive 4 new columns
    df['Sites'] = df.apply(mentioned_site, axis=1)
    df['Issues'] = df.apply(mentioned_issue, axis=1)
    df['Components'] = df.apply(mentioned_component, axis=1)
    df['Processed Feedback'] = df.apply(apply_nlp, axis=1)
    # scrfew it store as string

    # finally output the cleaned data to a CSV
    df.to_csv('output_pipeline.csv', encoding='ISO-8859-1')
    print("Outputted cleaned data to output_pipeline.csv")


run_pipeline(rf.filePath(rf.SITES), rf.filePath(rf.ORIGINAL_INPUT_DATA), 5000)
