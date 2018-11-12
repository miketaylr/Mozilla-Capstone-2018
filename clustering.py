# IMPORTS
from whoosh import index, writing, scoring
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import *
from whoosh.qparser import QueryParser
import os.path
from pathlib import Path
import tempfile
import subprocess
import csv
import codecs
import pandas as pd
import re
import os.path
import nltk as nltk
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn import preprocessing
import transformations as trafo
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster import hierarchy
import random
import seaborn as sns; sns.set()
import pprint
import referenceFiles as rf
import collections
from sklearn.feature_extraction import text


# SETTINGS - Paths
DIRECTORY = ""
OUTPUT_SPAM_REMOVAL = rf.filePath(rf.OUTPUT_SPAM_REMOVAL)
SITES = rf.filePath(rf.SITES)

# TODO: get a directory for the new spam removal output


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
            sf_output = row[columnToIndex]  # i.e. "sf_output" or "Negative Feedback"
            if sf_output != "" and isinstance(sf_output, str):
                neg_feedback = row[csvColumnName]
                response_id = row['Response ID']
                writer.update_document(index=str(i), sf_output=sf_output, negative_feedback=neg_feedback, response_id = response_id)
            i += 1
        writer.commit()

def getSitesList():
    sites = pd.read_csv(SITES, usecols=['Domains', 'Brand'])
    # , skiprows = 50, nrows = 25
    # display(sites)
    siteList = list(sites.values.flatten())

    # remove commas ('salesforce.com, force.com')
    for site in siteList:
        if ',' in site:
            siteList += site.split(',')

    siteList = [site.strip('.*').lower() for site in list(filter(lambda site: ',' not in site, siteList))]
    return siteList

def createNormalizedMatrix(siteList):
    # Create Reader to read in csv file after spam removal, read in the column from:
    schema = Schema(index=ID(stored=True),
                    response_id=ID(stored=True),
                    sf_output=TEXT(stored=True),
                    negative_feedback=TEXT(stored=True))
    indexToImport = createIndex(schema)
    # TODO: put indexToImport into dataframe instead of going through whoosh
    addFilesToIndex(indexToImport, OUTPUT_SPAM_REMOVAL, "Negative Feedback", "sf_output")
    myReader = indexToImport.reader()
    print("Index is empty?", indexToImport.is_empty())
    print("Number of indexed files:", indexToImport.doc_count())

    # If you want to see top 25 docs, print them below:
    # [(docnum, doc_dict) for (docnum, doc_dict) in myReader.iter_docs()][0:25]

    # Index all words in feedback from csv
    all_words = [term for term in myReader.field_terms("sf_output")]
    # Get term freq for specific term i.e. android
    # print(myReader.frequency("cell_content", "android"))
    # 1000 most distinctive terms according by TF-IDF score
    mostDistinctiveWords = [term.decode("ISO-8859-1") for (score, term) in
                            myReader.most_distinctive_terms("sf_output", 1000)]
    # 1000 most frequent words
    mostFrequentWords = [term.decode("ISO-8859-1") for (frequency, term) in
                         myReader.most_frequent_terms("sf_output", 1000)]

    wordVectorList = mostFrequentWords
    wordVectorList = [x for x in wordVectorList if x not in siteList]
    wordVectorList.remove('mozilla')
    wordVectorList.remove('firefox')
    print('Word List Length', len(wordVectorList))

    # Create a binary encoding of dataset based on the selected features (X)
    # Go through each document --> tokenize that single document --> compare with total word list

    tokenizer = RegexpTokenizer(r'\w+')
    df_rows = []
    word_list = wordVectorList

    # TODO: redundant code, remove this
    with codecs.open(OUTPUT_SPAM_REMOVAL, "r", "ISO-8859-1") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for i, row in enumerate(csvreader):
            value = row["sf_output"]
            if row["Negative Feedback"] != "" and isinstance(value, str): # Only looking for feedback that filled in negative feedback
                file_words = tokenizer.tokenize(value)
                df_rows.append([1 if word in file_words else 0 for word in word_list])
        X = pd.DataFrame(df_rows, columns=word_list)
        # convert to numpy array
        data = np.array(df_rows)
        # length normalization
        X_norm = preprocessing.normalize(data, norm='l2')
        rcOfX = X.shape
        # (r,c) = X.shape
    return X_norm, rcOfX[0], myReader


def kMeansClustering(X, numOfRows, myReader):
    # Run k-means
    # Setting number of clusters to hopefully split comments into sets of avg(10 FB comments)
    num_clusters = numOfRows / 10
    num_clusters = 20
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

    # NO NEED TO RUN IN FULL - START
    # Finding the best k - Elbow Method
    # k_average_inertia = []
    # local_inertia = []
    # random_states = []
    # num_tests = 5
    # # Get random nums to change random state, same for each k
    # for i in range(num_tests):
    #     state = random.randint(1, 1000)
    #     random_states.append(state)
    # # Run for k from 1-100, 5 runs each, then plot
    # for i in range(100):
    #     for state in random_states:
    #         # Number of clusters
    #         kmeans = KMeans(n_clusters=i + 1, random_state=state)
    #         # Fitting the input data
    #         kmeans = kmeans.fit(X_normalized)
    #         # Calculate inertia (within-cluster sum-of-squares)
    #         local_inertia.append(kmeans.inertia_)
    #     k_average_inertia.append(sum(local_inertia) / len(local_inertia))
    #     local_inertia = []
    #     print(i + 1, end=" ")
    # # Find the bend in this plot
    # plt.close() # To start a new with a fresh graph
    # plt.plot(k_average_inertia)
    # NO NEED TO RUN IN FULL - END

    # Hardcoded results for the Elbow Method
    hardcoded_k_average_inertia = [996.120161061241,
                                   971.7098230964491,
                                   952.7861704492907,
                                   940.0619752427674,
                                   930.3657061792022,
                                   918.9592073457455,
                                   913.3144481223687,
                                   904.4124416003946,
                                   899.1722839754657,
                                   892.5348899721223,
                                   886.2081548665992,
                                   882.3200576031841,
                                   876.2561460493231,
                                   873.3579443954989,
                                   872.4941626903143,
                                   860.9788965520158,
                                   857.7256214194788,
                                   858.3513261388655,
                                   849.9107789756945,
                                   863.6939208114576,
                                   849.2037298957055,
                                   857.5146014551663,
                                   841.618518098966,
                                   838.8230763546278,
                                   854.3754492848997,
                                   838.8054622193555,
                                   838.8318530622613,
                                   841.0033798673194,
                                   833.2005084255725,
                                   834.2733973798707,
                                   834.1165654175222,
                                   827.8246892480283,
                                   836.952528229488,
                                   825.3197614813049,
                                   820.8049740765731,
                                   823.9590905268827,
                                   828.0031583721487,
                                   819.973746683602,
                                   816.0232287008629,
                                   817.9730221219867,
                                   820.4044853040772,
                                   814.288086508465,
                                   811.6128144203527,
                                   813.0555432612409,
                                   807.676283599881,
                                   805.7903274017217,
                                   804.3946283001492,
                                   803.1046028442654,
                                   799.5363026790563,
                                   800.2921945062519,
                                   799.9831930769775,
                                   795.2297044615738,
                                   789.9335978665301,
                                   791.234355887476,
                                   787.3977369202623,
                                   785.6841859244969,
                                   783.6128876653244,
                                   783.1338156363867,
                                   781.3833871560394,
                                   779.36039738337,
                                   776.1094714925133,
                                   777.4673197143093,
                                   774.7611853114802,
                                   775.4384916877741,
                                   774.2833951321915,
                                   772.3287205585614,
                                   768.3819564072758,
                                   767.067666157384,
                                   765.1877408787177,
                                   766.2524704529731,
                                   764.8723960514769,
                                   762.2987814004226,
                                   763.2951371245215,
                                   760.7149203386532,
                                   758.0283566804861,
                                   759.9373693631244,
                                   755.2356418617525,
                                   754.9851550722124,
                                   755.6487467539771,
                                   753.9849972170422,
                                   753.2065379394986,
                                   752.0247411403313,
                                   750.8518415434044,
                                   747.5611285933307,
                                   746.5314162043006,
                                   746.9835885662785,
                                   744.1933443536657,
                                   744.2392574129924,
                                   740.090415931636,
                                   740.3859578173405,
                                   739.4696035101149,
                                   740.4889528631908,
                                   734.9103664672026,
                                   738.5949419195221,
                                   737.2100816822667,
                                   733.1439525352407,
                                   735.0596370040473,
                                   734.0226564464822,
                                   730.1183747156763,
                                   731.249786453746]
    plt.plot(hardcoded_k_average_inertia)

    # Show the num of docs/FB in each cluster, show the docs closest to centroid
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    clusterCount = dict(zip(unique, counts))
    plt.close() # To start a new with a fresh graph
    plt.bar(clusterCount.keys(), clusterCount.values(), color='g')
    plt.savefig("clusterCountBarGraph.png")
    # counts
    # Read in the actual feedback at this time
    # schema = Schema(index=ID(stored=True),
    #                 cell_content=TEXT(stored=True))
    # indexToImport = createIndex(schema)
    # # TODO: put indexToImport into dataframe instead of going through whoosh
    # addFilesToIndex(indexToImport, OUTPUT_SPAM_REMOVAL, "Negative Feedback", "Negative Feedback")
    # myReader = indexToImport.reader()
    # print("Index is empty?", indexToImport.is_empty())
    # print("Number of indexed files:", indexToImport.doc_count())
    # # See top 5 vectors closest to cluster centroid for all clusters
    # for j in range(num_clusters):
    #     d = kmeans.transform(X)[:, j]
    #     ind = np.argsort(d)[::-1][:5]
    #     print("Cluster", j)
    #     print("Indices of top 5 documents:", ind)
    #     print(np.array([doc_dict for doc_dict in myReader.iter_docs()])[ind])
    # fb = []
    # for content in myReader.iter_docs():
    #     fb.append(content)

    # OTHER WORK - NOT USING
    # the distance to the j'th centroid for each point in an array X
    # d = kmeans.transform(X)[:, j]
    # ind = np.argsort(d)[::-1][:5]
    # print("Cluster", j)
    # print("Indices of top 5 documents:", ind)
    # print(np.array([doc_dict for doc_dict in myReader.iter_docs()])[ind])
    # # X[ind] # What is this?
    # # SPECTURAL CLUSTERING
    # # note: df_rows not imported from first function above (createNormalizedMatrix)
    # from sklearn.cluster import SpectralClustering
    # from sklearn.metrics.pairwise import pairwise_distances
    # similarity_matrix = 1 - pairwise_distances(df_rows, metric='cosine')
    # cosineScores = pd.DataFrame(similarity_matrix)
    # clusters = SpectralClustering(n_clusters = 5, affinity = 'precomputed').fit(cosineScores)

    # TODO: need to decide what to return here for our clustering
    return labelsAsNums, kmeans, num_clusters, X #, fb


def visualizeSpectural():
    # ### FOR SPECTURAL CLUSTERING
    # def fit_and_plot(algorithm,title):
    #     col = ['bo','ro','co', 'mo','ko']
    #     algorithm.fit(X)
    #     n_clusters = algorithm.n_clusters
    #     lab = algorithm.labels_
    #     reds = lab == 0
    #     blues = lab == 1
    #     for jj in range(n_clusters):
    #         plt.plot(X[lab == jj, 0], X[lab == jj, 1], col[jj])
    #     plt.xlabel("$x_1$")
    #     plt.ylabel("$x_2$")
    #     plt.title(title)
    #     plt.axes().set_aspect('equal')
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(clusters.get_params())
    # fit_and_plot(clusters,"Spectral clustering on two circles")
    # y_kmeans = kmeans.predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    return


def labelClustersWKeywords(labels, myReader, kmeans, num_clusters, X):
        # # Get the key features (in our case, words) for each cluster
        # for j in range(num_clusters):
        #     relevantFB = ["",""]
        #     FBnum = 0
        #     for label in labels:
        #         if label == j:
        #             thisTuple = fb[FBnum]
        #             relevantFB.append(thisTuple[1])
        #         FBnum = FBnum + 1
        #     df = pd.DataFrame(relevantFB)
        #     vectorizer = CountVectorizer(min_df=1, stop_words='english')
        #     featuresCounted = vectorizer.fit_transform(d.get('cell_content') for d in df[1])
        #     print(vectorizer.get_feature_names())
        #     print(featuresCounted.toarray())
        #     print("hey")

    # ### TODO: FIXXXXXXXXX BRUUUUHHHHHHHHHHHHHHHH
    # Pulling out key words to label cluster / understand what is in each cluster
    # pull out documents of each cluster --> tf idf for key words
    # test_cluster = 12
    top_features_list = []

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

def labelClustersWithKeyPhrases(labels, myReader, kmeans, num_clusters, X):
    for cluster in num_clusters:
        print('Cluster', cluster)

def clusterPerformanceMetrics(labels, myReader, num_clusters):
    # NOTE: CSV MUST BE SORTED BY ID AND SAVED THAT WAY
    sr = pd.read_csv('data/output_clusters_defined.csv')

    for cluster in range(num_clusters):
        clusterIndices = [index for index, clusterNum in enumerate(labels) if clusterNum == cluster]
        docIndices = [int(doc_dict['index']) for (docnum, doc_dict) in myReader.iter_docs() if docnum in clusterIndices]
        response_ids = [int(doc_dict['response_id']) for (docnum, doc_dict) in myReader.iter_docs() if
                        docnum in clusterIndices]

        manual_cluster_counts = sr.loc[sr['Response ID'].isin(response_ids)]['manual_clusters'].fillna('-1').astype(int).value_counts()
        print("Cluster", cluster)
        print(manual_cluster_counts)

    return



# run
print('We startin')
siteList = getSitesList()
X_norm, numOfFB, readerForFullFB = createNormalizedMatrix(siteList)
labels, kmeans, num_clusters, X = kMeansClustering (X_norm, numOfFB, readerForFullFB)
# visualizeSpectural
feature_names_df = labelClustersWKeywords(labels, readerForFullFB, kmeans, num_clusters, X)
print('Top 5 words in each cluster:')
print(feature_names_df)
clusterPerformanceMetrics(labels, readerForFullFB, num_clusters)
print('We done.')





