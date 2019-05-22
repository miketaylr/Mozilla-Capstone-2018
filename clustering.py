# IMPORTS
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
import os.path
import tempfile
import csv
import codecs
import pandas as pd
import re
import os.path
import nltk as nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import referenceFiles as rf
import collections
from nltk.tag import PerceptronTagger
from sklearn.cluster import SpectralClustering
from datetime import datetime as datetime
from sklearn.cluster import AgglomerativeClustering
import os
import time
import math


# SETTINGS - Paths
DIRECTORY = ""
OUTPUT_SPAM_REMOVAL = rf.filePath(rf.OUTPUT_SPAM_REMOVAL)
SITES = rf.filePath(rf.SITES)

pd.set_option('display.max_columns',10)

g_custom_stop_words = ['mozilla','firefox']


def createIndex(schema):
    # Generate a temporary directory for the index
    indexDir = tempfile.mkdtemp()
    # create and return the index
    return index.create_in(indexDir, schema)


def addFilesToIndex(indexObj, csvPath, csvColumnName, columnToIndex):
    # open writer
    writer = indexObj.writer()
    # open csv
    with codecs.open(csvPath, "r", "ISO-8859-1") as csvfile:
        # create csv reader object
        csvreader = csv.DictReader(csvfile)
        # instantiate index count
        i = 0
        # read each row in file
        for row in csvreader:
            sf_output = row[columnToIndex]
            neg_feedback = row[csvColumnName]
            if sf_output != "" and isinstance(sf_output, str) and neg_feedback != "":
                response_id = row['Response ID']
                writer.update_document(index=str(i), sf_output=sf_output, negative_feedback=neg_feedback, response_id = response_id)
            i += 1
        writer.commit()


def getSitesList():
    sites = pd.read_csv(SITES, usecols=['Domains', 'Brand'])
    # display(sites)
    siteList = list(sites.values.flatten())

    # remove commas ('salesforce.com, force.com')
    for site in siteList:
        if ',' in site:
            siteList += site.split(',')

    siteList = [site.strip('.*').lower() for site in list(filter(lambda site: ',' not in site, siteList))]
    return siteList


def createNormalizedMatrix(file, custom_stop_words = g_custom_stop_words): #not the most efficient. we index everything in feedback column and pull out all unique words and put it into a corpus
    # Create Reader to read in csv file after spam removal, read in the column from:
    schema = Schema(index=ID(stored=True),
                    response_id=ID(stored=True),
                    sf_output=TEXT(stored=True),
                    negative_feedback=TEXT(stored=True))
    indexToImport = createIndex(schema)
    # TODO: put indexToImport into dataframe instead of going through whoosh
    addFilesToIndex(indexToImport, file, "Feedback", "sf_output")
    myReader = indexToImport.reader()
    print("Index is empty?", indexToImport.is_empty())
    print("Number of indexed files:", indexToImport.doc_count())

    mostDistinctiveWords = [term.decode("ISO-8859-1") for (score, term) in
                            myReader.most_distinctive_terms("sf_output", 500) if term not in ENGLISH_STOP_WORDS]
    # 1000 most frequent words
    mostFrequentWords = [term.decode("ISO-8859-1") for (frequency, term) in
                         myReader.most_frequent_terms("sf_output", 500) if term not in ENGLISH_STOP_WORDS]

    # change to take 500 instead of 1000 above, and use a combination of distinct and frequent words
    wordVectorList = mostDistinctiveWords + mostFrequentWords
    # wordVectorList = top1000MIWords

    # wordVectorList = [x for x in wordVectorList if x not in siteList]
    for word in custom_stop_words:
        if word in wordVectorList:
            wordVectorList.remove(word)
    # if 'mozilla' in wordVectorList:
    #     wordVectorList.remove('mozilla')
    # if 'firefox' in wordVectorList:
    #     wordVectorList.remove('firefox')
    print('Word List Length', len(wordVectorList))

    # Create a binary encoding of dataset based on the selected features (X)
    # Go through each document --> tokenize that single document --> compare with total word list
    tokenizer = RegexpTokenizer(r'\w+')
    df_rows = []
    feedback_length = []
    time_difference = []
    word_list = wordVectorList

    # while loop below is indexing, goes through all feedback, binary ecnoding (should we switch to freq enc?), does it contain this feature? (1 or 0)
    with codecs.open(file, "r", "ISO-8859-1") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for i, row in enumerate(csvreader):
            value = row["sf_output"]
            if row["Feedback"] != "" and isinstance(value, str): # TODO: change to just look at negative feedback
                file_words = tokenizer.tokenize(value)
                df_rows.append([1 if word in file_words else 0 for word in word_list])
                if isinstance(row["Feedback"], str):
                    feedback_length.append(len(row["Feedback"]))
                else:
                    feedback_length.append(0)
                # https://stats.stackexchange.com/questions/105959/best-way-to-turn-a-date-into-a-numerical-feature
                if isinstance(row["Date Submitted"], str):
                    reference_now = datetime.now()
                    date_time_str = row["Date Submitted"]
                    datetime_formatted = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                    difference = (reference_now-datetime_formatted)
                    time_difference.append(int(difference.total_seconds()))
        X_raw = pd.DataFrame(df_rows, columns=word_list)
        length_feature = pd.DataFrame({'fb_length': np.array(feedback_length)})
        time_feature = pd.DataFrame({'time_to_now': np.array(time_difference)})
        X_and_length = X_raw.join(length_feature)
        X = X_and_length.join(time_feature)
        print(list(X))
        # convert to numpy array
        data = np.array(df_rows)
        # length normalization
        X_norm_raw = preprocessing.normalize(data, norm='l2')
        length_feature_norm = preprocessing.normalize(length_feature.values, norm='l1')
        time_feature_norm = preprocessing.normalize(time_feature.values, norm='l1')
        # TODO: fix the normalization of the length feature and time feature
        X_and_length_norm = np.append(X_norm_raw, length_feature_norm, axis=1)
        # X_norm = np.append(X_and_length_norm, time_feature_norm, axis=1)
        X_norm = np.append(X_and_length_norm, time_feature_norm, axis=1)
        # ignore time and length 
        X_norm = X_raw
        rcOfX = X_norm.shape
    return X_norm, rcOfX[0], myReader


def kMeansClustering(X, numOfRows, num_clusters):
    # Run k-means
    # Setting number of clusters to hopefully split comments into sets of avg(10 FB comments)
    # TODO: make this (^) more robust / logical
    kmeans = KMeans(n_clusters=num_clusters, random_state = 40)
    # Fitting the input data
    kmeans = kmeans.fit(X)
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # Centroid values
    centroids = kmeans.cluster_centers_
    labelsAsNums = kmeans.labels_
    print(centroids)
    print(labelsAsNums)

    # TODO: need to decide what to return here for our clustering
    return labelsAsNums, kmeans, X #, fb


def labelClustersWKeywords(labels, myReader, num_clusters):
    top_features_list = []
    print(myReader)

    for cluster in range(num_clusters):
        indices = [index for index, clusterNum in enumerate(labels) if clusterNum == cluster] # indices of documents in cluster
        clusterCorpus = [doc_dict['negative_feedback'] for (docnum, doc_dict) in myReader.iter_docs() if docnum in indices] # documents in cluster

        custom_stop_words = ENGLISH_STOP_WORDS.union(["firefox"])
        vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
        X_tf = vectorizer.fit_transform(clusterCorpus)
        response = vectorizer.transform(clusterCorpus)
        feature_names = vectorizer.get_feature_names()

        top_n = 5
        feature_name_occurences = np.nonzero(response.toarray())[1]
        most_common_n = collections.Counter(feature_name_occurences).most_common(top_n)
        top_features = [feature_names[feature[0]] for feature in most_common_n]
        top_features_list.append(top_features)

    feature_names_df = pd.DataFrame(top_features_list, columns=['1', '2', '3', '4', '5'])
    return feature_names_df


def labelClustersWithKeyPhrases(labels, myReader, num_clusters, n):
    top_features_list = []

    tagger = PerceptronTagger()
    pos_tag = tagger.tag
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    # Create phrase tree
    chunker = nltk.RegexpParser(grammar)

    stop = ENGLISH_STOP_WORDS

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

    def flatten(npTokenList):
        finalList = []
        for phrase in npTokenList:
            token = ''
            for word in phrase:
                token += word + ' '
            finalList.append(token.rstrip())
        return finalList

    for cluster in range(num_clusters):
        indices = [index for index, clusterNum in enumerate(labels) if clusterNum == cluster] # indices of documents in cluster
        clusterCorpus = [doc_dict['negative_feedback'] for (docnum, doc_dict) in myReader.iter_docs() if docnum in indices] #
        clusterCorpus = ' '.join(clusterCorpus)

        counter = Counter()
        counter.update(flatten([word
                                for word
                                in get_terms(chunker.parse(pos_tag(re.findall(r'\w+', clusterCorpus))))
                                ]))

        most_common_n = counter.most_common(n)

        top_features = [feature[0] for feature in most_common_n]
        top_features_list.append(top_features)

    feature_names_df = pd.DataFrame(top_features_list, columns=['1', '2', '3', '4', '5'])

    return feature_names_df


def clusterPerformanceMetrics(labels, myReader, num_clusters):
    # NOTE: CSV MUST BE SORTED BY ID AND SAVED THAT WAY
    sr = pd.read_csv('data/output_clusters_defined.csv')
    max = 0
    total_count = 0
    clusterCountSeries = pd.Series([])
    clusterDesc = pd.read_csv('./data/manual_cluster_descriptions.csv')
    categoryDict = pd.Series(clusterDesc.description.values, index=clusterDesc.clusters_types).to_dict()
    clusterGroupDF = {}

    for cluster in range(num_clusters):
        clusterIndices = [index for index, clusterNum in enumerate(labels) if clusterNum == cluster]
        docIndices = [int(doc_dict['index']) for (docnum, doc_dict) in myReader.iter_docs() if docnum in clusterIndices]
        response_ids = [int(doc_dict['response_id']) for (docnum, doc_dict) in myReader.iter_docs() if
                        docnum in clusterIndices]

        clusterData = sr.loc[sr['Response ID'].isin(response_ids)].groupby(['manual_clusters'])
        clusterGroupDict = {}

        for key in clusterData.groups.keys():
            phraseKey = categoryDict[int(key)]
            clusterGroupDict[phraseKey] = list(clusterData.get_group(key)['Negative Feedback'])

        clusterGroupDF['Cluster ' + str(cluster)] = str(clusterGroupDict)

        manual_cluster_counts = sr.loc[sr['Response ID'].isin(response_ids)]['manual_clusters'].fillna('-1').astype(int).value_counts()
        max += manual_cluster_counts.nlargest(1).iloc[0]
        total_count += sr.loc[sr['Response ID'].isin(response_ids)]['manual_clusters'].fillna('-1').astype(int).count()
        # print("Cluster", cluster)
        # print(manual_cluster_counts)
        # frames.append(dict(manual_cluster_counts))
        clusterCountSeries = pd.concat([clusterCountSeries, manual_cluster_counts.rename('Cluster ' + str(cluster))], axis=1)

    # pd.concat(frames, keys=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    purity = max/total_count
    clusterCountSeries = clusterCountSeries.drop(0, 1)
    clusterGroupDF = pd.DataFrame.from_records([clusterGroupDF], index = ['Docs'])

    return purity, clusterCountSeries, clusterGroupDF


def spectralClustering(X, num_clusters):
    clusters = SpectralClustering(n_clusters = num_clusters, affinity='cosine', random_state=40).fit(X)
    labelsAsNums = clusters.labels_
    return labelsAsNums, clusters, X


def hierarchicalClustering(X, num_clusters, linkage):
    clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage=linkage).fit(X)
    labelsAsNums = clusters.labels_
    return labelsAsNums, clusters, X


def purityElbowGraph(X_norm, numOfFB, readerForFullFB):
    # Purity elbow graph
    purity = []
    clusterSize = []
    for n in range(20, 1000, 40):
        # labels, kmeans, X = kMeansClustering(X_norm, numOfFB, n)
        # labels, spectral, X = spectralClustering(X_norm, n)
        labels, hierarchical, X = hierarchicalClustering(X_norm, n, 'single')
        clusterPurity, clusterCountSeries, clusterGroupDF = clusterPerformanceMetrics(labels, readerForFullFB, n)
        # print(clusterPurity)
        purity.append(clusterPurity)
        clusterSize.append(n)
        print(n)
    plt.plot(clusterSize, purity)
    plt.title('Purity vs Number of Clusters using Hierarchical: Single')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Purity')
    # DO NOT RUN THIS AGAIN IT TAKES FOREVER
    plt.savefig("purityElbowMethodHierarchicalSingle1000.png")

def barGraphVisualization(labels, kmeans, X, top_words, top_phrases, clusterCountSeries, clusterGroupDF):
    top_words_combined, top_phrases_combined = condenseTopWordsPhrases(top_words, top_phrases)

    countsWords = clusterCountSeries.append(top_words_combined, ignore_index=False)
    countsWordsPhrases = countsWords.append(top_phrases_combined, ignore_index = False)
    countsWordsPhrasesDocs = countsWordsPhrases.append(clusterGroupDF, ignore_index = False)

    return countsWordsPhrasesDocs

def condenseTopWordsPhrases(top_words, top_phrases):
    top_words_list = top_words.applymap(lambda x: [x] if pd.notnull(x) else []).sum(1).tolist()

    top_words['combined'] = [", ".join(item) for item in top_words_list]
    top_words_combined = top_words['combined'].rename(lambda x: 'Cluster ' + str(x))
    top_words_combined.name = 'Words'

    top_phrases_list = top_phrases.applymap(lambda x: [x] if pd.notnull(x) else []).sum(1).tolist()

    top_phrases['combined'] = [", ".join(item) for item in top_phrases_list]
    top_phrases_combined = top_phrases['combined'].rename(lambda x: 'Cluster ' + str(x))
    top_phrases_combined.name = 'Phrases'
    return top_words_combined, top_phrases_combined

def runVis(num_clusters):
    X_norm, numOfFB, readerForFullFB = createNormalizedMatrix(OUTPUT_SPAM_REMOVAL)

    labels, kmeans, X = kMeansClustering(X_norm, numOfFB, num_clusters)
    feature_names_df_kmeans = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    feature_phrases_df_kmeans = labelClustersWithKeyPhrases(labels, readerForFullFB, num_clusters, 5)
    purity, clusterCountSeries, clusterGroupDF = clusterPerformanceMetrics(labels, readerForFullFB, num_clusters)

    visDf = barGraphVisualization(labels, kmeans, X, feature_names_df_kmeans, feature_phrases_df_kmeans, clusterCountSeries, clusterGroupDF)
    return visDf


def run(): # don't delete this, but we're not using it, use run drilldown()
    # run
    print('We startin')
    siteList = getSitesList()
    num_clusters = 100
    X_norm, numOfFB, readerForFullFB = createNormalizedMatrix(OUTPUT_SPAM_REMOVAL)

    # # K Means
    print(" --- K MEANS ---")
    labels, kmeans, X = kMeansClustering (X_norm, numOfFB, num_clusters)
    feature_names_df_kmeans = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    feature_phrases_df_kmeans = labelClustersWithKeyPhrases(labels, readerForFullFB, num_clusters, 5)
    print('Top 5 words in each cluster:')
    print(feature_names_df_kmeans)
    print('Top 5 phrases in each cluster:')
    print(feature_phrases_df_kmeans)

    # TODO: currently not working because of csv mismatch
    # purity, clusterCountSeries, clusterGroupDF = clusterPerformanceMetrics(labels, readerForFullFB, num_clusters)
    # print('Purity', purity)

    # # # Spectral
    # print(" --- SPECTRAL ---")
    # labels, spectral, X = spectralClustering(X_norm, num_clusters)
    # feature_names_df_spectral = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    # feature_phrases_df_spectral = labelClustersWithKeyPhrases(labels, readerForFullFB, num_clusters, 5)
    # print('Top 5 words in each cluster:')
    # print(feature_names_df_spectral)
    # print('Top 5 phrases in each cluster:')
    # print(feature_phrases_df_spectral)
    # purity, clusterCountSeries = clusterPerformanceMetrics(labels, readerForFullFB, num_clusters)
    # print('Purity', purity)

    # # # Hierarchical
    # NOTE: currently not working because there are likely empty values??
    # print(" --- HIERARCHICAL ---")
    # labels, hierarchicalClusters, X = hierarchicalClustering(X_norm, num_clusters)
    # feature_names_df_spectral = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    # print('Top 5 words in each cluster:')
    # print(feature_names_df_spectral)
    # purity, clusterCountSeries = clusterPerformanceMetrics(labels, readerForFullFB, num_clusters)
    # print('Purity', purity)

    # Graph Purity vs num of clusters (for performance metrics purposes)
    # purityElbowGraph(X_norm, numOfFB, readerForFullFB)

    print('We done.')
    return

def doKMeans(X_norm, numOfFB, readerForFullFB, num_clusters, df):
    labels, clusters, X = kMeansClustering(X_norm, numOfFB, num_clusters) # if want to switch to spectral/hierarchical, switch it in here
    feature_names_df_kmeans = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    feature_phrases_df_kmeans = labelClustersWithKeyPhrases(labels, readerForFullFB, num_clusters, 5)
    
    unique, counts = np.unique(labels, return_counts = True)
    counts = pd.Series(counts).rename(lambda x: 'Cluster ' + str(x))
    counts.name = 'Count'

    ids = pd.Series([df.iloc[np.where(labels == n)[0].tolist()]['Response ID'].tolist() for n in range(num_clusters)]).rename(lambda x: 'Cluster ' + str(x))
    ids.name = 'Response IDs'

    top_words_combined, top_phrases_combined = condenseTopWordsPhrases(feature_names_df_kmeans, feature_phrases_df_kmeans)
    final = pd.concat([counts, ids, top_words_combined, top_phrases_combined], axis=1)
    return final

def doSpectral(X_norm, numOfFB, readerForFullFB, num_clusters, df):
    clusters = SpectralClustering(n_clusters = num_clusters, affinity='cosine', random_state=40).fit(X_norm)
    labels = clusters.labels_    

    feature_names_df_kmeans = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    feature_phrases_df_kmeans = labelClustersWithKeyPhrases(labels, readerForFullFB, num_clusters, 5)
    
    unique, counts = np.unique(labels, return_counts = True)
    counts = pd.Series(counts).rename(lambda x: 'Cluster ' + str(x))
    counts.name = 'Count'

    ids = pd.Series([df.iloc[np.where(labels == n)[0].tolist()]['Response ID'].tolist() for n in range(num_clusters)]).rename(lambda x: 'Cluster ' + str(x))
    ids.name = 'Response IDs'

    top_words_combined, top_phrases_combined = condenseTopWordsPhrases(feature_names_df_kmeans, feature_phrases_df_kmeans)
    final = pd.concat([counts, ids, top_words_combined, top_phrases_combined], axis=1)
    return final

def doAgglomerative(X_norm, numOfFB, readerForFullFB, num_clusters, df):
    # clusters = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='single').fit(X_norm)
    labels = clusters.labels_

    feature_names_df_kmeans = labelClustersWKeywords(labels, readerForFullFB, num_clusters)
    feature_phrases_df_kmeans = labelClustersWithKeyPhrases(labels, readerForFullFB, num_clusters, 5)
    
    unique, counts = np.unique(labels, return_counts = True)
    counts = pd.Series(counts).rename(lambda x: 'Cluster ' + str(x))
    counts.name = 'Count'

    ids = pd.Series([df.iloc[np.where(labels == n)[0].tolist()]['Response ID'].tolist() for n in range(num_clusters)]).rename(lambda x: 'Cluster ' + str(x))
    ids.name = 'Response IDs'

    top_words_combined, top_phrases_combined = condenseTopWordsPhrases(feature_names_df_kmeans, feature_phrases_df_kmeans)
    final = pd.concat([counts, ids, top_words_combined, top_phrases_combined], axis=1)
    return final


def runDrilldown(df): #this is integrated into dash interface, everything that isn't called here we don't use START HERE, but if we swap to hierarchical/spectral, uncomment the functions related to those DONT DELETE ANYTHING
    # Create temp csv using timestamp as name
    filename = 'data/' + str(round(time.time())) + '.csv' #the file that nicole is passing through, she names them based on time (in the data folder you see the seconds.csv)
    df.to_csv(filename)

    # Create index based off of csv -- this takes a long time, so we just pass in the reader
    X_norm, numOfFB, readerForFullFB = createNormalizedMatrix(filename)

    # Delete temp csv FIX
    os.remove(filename)
    print(len(df))
    # 10 docs per cluster; ceil because if less than 10 docs, then outputs 1 cluster
    num_clusters = math.ceil(len(df)/5)    
    
    final_KMeans = doKMeans(X_norm, numOfFB, readerForFullFB, num_clusters, df)

    final_Spectral = doSpectral(X_norm, numOfFB, readerForFullFB, num_clusters, df)

    final = final_Spectral
    return final


# runDrilldown(pd.read_csv("./data/output_spam_removed.csv", encoding ="ISO-8859-1"))
# run()