import os

DIRECTORY = "data/"

SITES = 'Top Sites for Report Analysis.csv'
BRAND_KEYWORDS = 'brand-keywords.csv'
ISSUE_KEYWORDS = 'issue-keywords.csv'
COMPONENT_KEYWORDS = 'component-keywords.csv'
COUNTRY_LIST = 'countries.csv'

SPAM_LABELLED = 'output_spam_labelled.csv'
ORIGINAL_INPUT_DATA = '20190211135347-SurveyExport.csv'
INPUT_DATA_1 = '20181001120735-SurveyExport.csv'

OUTPUT_PIPELINE = 'output_pipeline.csv'
OUTPUT_SPAM_REMOVAL = 'output_spam_removed.csv'
OUTPUT_CATEGORIZATION = 'output_categorization.csv'
OUTPUT_CLUSTER = 'output_clustering.csv'

TOP_WORDS = 'topwords.txt'

def filePath(file):
    return os.path.join(DIRECTORY, file)