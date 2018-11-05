import os

DIRECTORY = "data/"

SITES = 'Top Sites for Report Analysis.csv'
SPAM_LABELLED = 'output_spam_labelled.csv'
ORIGINAL_INPUT_DATA = '20181001120735-SurveyExport.csv'
INPUT_DATA_1 = '20181103112740-SurveyExport.csv'

OUTPUT_PIPELINE = 'output_pipeline.csv'
OUTPUT_SPAM_REMOVAL = 'output_spam_removed.csv'
OUTPUT_CATEGORIZATION = 'output_categorization.csv'
OUTPUT_CLUSTER = 'output_clustering.csv'

def filePath(file):
    return os.path.join(DIRECTORY, file)