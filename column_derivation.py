#Purpose of this file is to derive columns to be used in the final cleaned dataset
# as part of the pipeline process
import pandas as pd


def apply_nlp(row):
    combined = row['Positive Feedback'] + row['Negative Feedback']
    filtered = [token.text for token in stmLwrFilter(combined)]
    return ','.join(set(filtered))  # turn to set temporarily to get unique values


# crude way of looking for mentioned site using the top 100 list. Need to add the regex to pick up wildcard sites
def mentioned_site(row):
    combined = row['Feedback'].lower() + ' ' + row['Relevant Site'].lower()
    # sites = [site.lower() for site in siteList if site.lower() in combined]
    urls = re.findall("https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+",
                      combined)  # NEED TO IMPROVE REGEX TO PICK UP MORE SITES
    if len(urls) == 0:
        urls = [row['Relevant Site'].lower()]

    sites = urls
    return ','.join(set(sites))

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
def evalSentences(sia, sentence):  # only one sentence
    # Instantiate an instance to access SentimentIntensityAnalyzer class
    sid = sia
    ss = sid.polarity_scores(sentence)  # run sentiment on the sentence
    return [ss['compound']]  # return the value of the compound sentiment


def derive_columns(data_frame, siteList):  # based on cols from data after cleaning

    data_frame['Feedback'] = data_frame['Positive Feedback'].map(str) + data_frame['Negative Feedback'].map(str)
    data_frame['Sites'] = data_frame.apply(mentioned_site, axis=1)

    # data_frame = data_frame.merge(df['Feedback'].apply(lambda s: pd.Series(
    #     {'Sites': re.findall(re.compile(siteList + '|https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'), s)})),
    #
    #                               left_index=True, right_index=True)
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
    # df['Processed Feedback'] = df.apply(apply_nlp, axis=1) not using this NLP anymore.


# def langid_language_filter(row):
#     combined = row['Positive Feedback'] + row['Negative Feedback']
#     language = [langid.classify(text)[0] for text in combined.lower()]
#     return language

# Initialize and derive 4 new columns
df = derive_columns(df)
