import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table_experiments as dte
import pandas as pd
import plotly.graph_objs as go
from clustering import runDrilldown
from datetime import datetime as datetime
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE, top_sites
from collections import Counter
import ast


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://fonts.googleapis.com/css?family=Montserrat:300,100']
external_scripts = ['https://code.jquery.com/jquery-3.2.1.min.js', 'https://d3js.org/d3.v4.min.js']


# Reading in data:
results_df = pd.read_csv("./data/output_pipeline.csv", encoding ="ISO-8859-1")
results2_df = pd.read_csv("./data/output_pipeline.csv", encoding="ISO-8859-1")
sr_df = pd.read_csv("./data/output_spam_removed.csv", encoding="ISO-8859-1")


search_df = results_df
for index, row in search_df.iterrows():
    if pd.isnull(row['Sites']):
        search_df.at[index, 'Sites'] = 'None Found'
df_geo = pd.read_csv('./data/output_countries.csv')
df1 = pd.read_csv('./data/Issues_Keywords_Clusters.csv', encoding='latin-1')
component_df = pd.read_csv('./data/component_graph_data.csv')
issue_df = pd.read_csv('./data/issue_graph_data.csv')
clusterDesc = pd.read_csv('./data/manual_cluster_descriptions.csv')
clusters_df = pd.read_csv('./data/output_clusters_defined.csv', usecols = ['Response ID', 'manual_clusters'])
global_site_modal_ids = []
global_comp_modal_ids = []
global_issue_modal_ids = []
global_top_selected_sites = []
global_other_selected_sites = []
global_geo_modal_ids = []
global_selected_geo = []
topSiteCloseCount=0
otherSiteCloseCount=0
compCloseCount=0
issueCloseCount=0
geoCloseCount=0


# GLOBALLY ADD DAY DIFFERENCE TO RESULTS DATAFRAME
reference = datetime(2018, 11, 19)
results2_df['Day Difference'] = (reference - pd.to_datetime(results2_df['Date Submitted'], format='%Y-%m-%d %H:%M:%S')).dt.days + 1

global_sentiment_average = results2_df['compound'].mean()
print(global_sentiment_average)
df_geo = df_geo.rename(columns={'COUNTRY': 'Country'})
geo_2week_df = df_geo[['Country']]
# Calculate daily average sentiment scores over the past 2 weeks
for num_days in range(14):
    past_x_days_df = results2_df[results2_df['Day Difference'] <= num_days + 1][['Country', 'compound']].groupby('Country', as_index=False).mean()
    past_x_days_df.columns = ['Country', num_days+1]
    geo_2week_df = pd.merge(geo_2week_df, past_x_days_df, on='Country', how='left')

one_week_compound_df = results2_df[results2_df['Day Difference'] <= 7][['Country', 'compound']]
one_week_compound_df.columns = ['Country', 'Sentiment_Week']
full_compound_df = results2_df[['Country', 'compound']]
full_compound_df.columns = ['Country', 'Sentiment_Full']

combined_compound_df = pd.merge(one_week_compound_df, full_compound_df, on='Country', how='inner')
combined_compound_df['Sentiment_Norm'] = combined_compound_df['Sentiment_Week'] - combined_compound_df['Sentiment_Full']
combined_compound_df['Sentiment_Norm_Global'] = combined_compound_df['Sentiment_Full'] - global_sentiment_average

df_geo = df_geo.drop('Sentiment', axis=1)
df_geo.columns = ['Country', 'Code']
df_geo.set_index('Country')

df_review_compound =  combined_compound_df.groupby('Country', as_index=False).mean()
df_review_compound = df_review_compound[['Country', 'Sentiment_Norm', 'Sentiment_Norm_Global', 'Sentiment_Week', 'Sentiment_Full']]
df_review_compound.set_index('Country')

df_geo_sentiment = pd.merge(df_review_compound, df_geo, on='Country', how='inner')
df_geo_sentiment = pd.merge(df_geo_sentiment, geo_2week_df, on='Country', how='inner')
df_geo_sentiment = df_geo_sentiment.drop('Sentiment_Week', axis = 1)


def updateGeoGraph(df, type, value):
    print(df)
    if type=='norm':
        sentiment = df[value] - df['Sentiment_Full']
    elif type=='globalNorm':
        sentiment = df['Sentiment_Norm_Global']
    else:
        sentiment = df[value]
    fig_geo = dict(data=[
            dict(
                type = 'choropleth',
                locations = df['Code'],
                z = sentiment,
                text = df['Country'],
                colorscale = [ [0.0, 'rgb(153,0,13)'], [0.1, 'rgb(203,24,29)'], [0.2, 'rgb(239,59,44)'], [0.3, 'rgb(251,106,74)'], [0.4, 'rgb(252,146,114)'], \
                               [0.5, 'rgb(252,187,161)'], [0.55, 'rgb(199,233,192)'], [0.65, 'rgb(161,217,155)'], [0.75, 'rgb(116,196,118)'], [0.85, 'rgb(65,171,93)'],  [0.95, 'rgb(35,139,69)'], [1,'rgb(0,109,44)']],
                autocolorscale = False,
                reversescale = False,
                marker = dict(
                    line = dict (
                        color = 'rgb(180,180,180)',
                        width = 0.7
                    ) ),
                colorbar = dict(
                    autotick = False,
                    tickprefix = '',
                    title = 'Global Sentiment Score'),
                    # color = '#D3D3D3'
        )],
        layout=dict(
            title = 'Global Sentiment Scores',
            titlefont=dict(
                family='Montserrat, Helvetica Neue, Helvetica, sans-serif',
                color='white'
            ),
            geo = dict(
                showframe = False,
                showcoastlines = False,
                projection = dict(
                    type = 'Mercator'
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
            font=dict(
                family='Montserrat, Helvetica Neue, Helvetica, sans-serif',
                size=12,
                color='white',
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig_geo


fig_geo = updateGeoGraph(df_geo_sentiment, '',7)


# Getting components and issues in string:
WORDS_TO_COMPONENT = {k:(map(lambda word: word.lower(), v)) for k, v in WORDS_TO_COMPONENT.items()}
WORDS_TO_ISSUE = {k:(map(lambda word: word.lower(), v)) for k, v in WORDS_TO_ISSUE.items()}

# Hardcoded Fake Data
arrayOfNames = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking']
arrayOfNamesWords = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking', 'Words']
arrayOfNamesDocs = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking', 'Docs']
numClusters = 50
traces = []
clusterNames = list(df1)
clusterNames.pop(0)
df1 = df1.set_index('Issue')
docs = df1.drop(arrayOfNamesWords, axis=0)
words = df1.drop(arrayOfNamesDocs, axis=0)
clusters = df1.drop(['Words', 'Docs'], axis=0)

categoryDict = pd.Series(clusterDesc.description.values, index=clusterDesc.clusters_types).to_dict()


# TIME CALCULATION
toggle_time_params = {
    'min': 0,
    'max': 14,
    'step': 1,
    'default': 7,
    'marks': {
        0: '',
        7: '1 Week',
        14: '2 Weeks'
    }
}


def initCompDF(results2_df, num_days_range = 14):
    date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
    date_filtered_df['Components'] = date_filtered_df['Components'].apply(
        lambda x: ast.literal_eval(x))  # gives warning but works, fix later

    component_df = pd.Series([])
    comp_response_id_map = dict()
    comp_day_response_id_map = dict()

    for day in range(num_days_range):
        day_df = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]
        if(day_df.empty):
            continue
        date = str(day_df['Date Submitted'].values[0]).split(' ')[0]
        # count docs with components
        new_comp_info = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
            lambda x: pd.Series(x).value_counts()).sum()
        # count docs with no assigned components
        if(0 in date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
                                       lambda x: len(x)).value_counts().index):
            new_comp_info = pd.concat([new_comp_info,
                                        date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
                                        lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])

        component_df = pd.concat([component_df, new_comp_info.rename(date)], axis=1)
        comp_response_id_map[date] = dict()
        comp_day_response_id_map[date] = []
        comps = new_comp_info.index.values
        for comp in comps:
            comp_response_id_map[date][comp] = []
        for index, row in day_df.iterrows():
            comp_day_response_id_map[date].append(row['Response ID'])
            for comp in row['Components']:
                comp_response_id_map[date][comp].append(
                    row['Response ID'])  # TODO: can use map functions to make this faster
            if len(row['Components']) == 0 and 'No Label' in comps:
                comp_response_id_map[date]['No Label'].append(row['Response ID'])

    component_df = component_df.fillna(0).astype(int).rename_axis('Components')
    component_df.drop([0], axis = 1, inplace = True)
    return component_df, comp_response_id_map, comp_day_response_id_map


def initIssueDF(results2_df, num_days_range = 14):
    date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
    date_filtered_df['Issues'] = date_filtered_df['Issues'].apply(lambda x: ast.literal_eval(x))

    issue_df = pd.Series([])
    issue_day_response_id_map = dict()
    issue_response_id_map = dict()

    for day in range(num_days_range):
        day_df = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]
        if(day_df.empty):
            continue
        date = str(day_df['Date Submitted'].values[0]).split(' ')[0]
        new_issue_info = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
            lambda x: pd.Series(x).value_counts()).sum()
        # count docs with no assigned components
        if(0 in date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
                                       lambda x: len(x)).value_counts().index):
            new_issue_info = pd.concat([new_issue_info,
                                        date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
                                        lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])

        issue_df = pd.concat([issue_df, new_issue_info.rename(date)], axis=1)

        issue_response_id_map[date] = dict()
        issue_day_response_id_map[date] = []

        issues = new_issue_info.index.values
        for issue in issues:
            issue_response_id_map[date][issue] = [];

        for index, row in day_df.iterrows():
            issue_day_response_id_map[date].append(row['Response ID'])
            for issue in row['Issues']:
                issue_response_id_map[date][issue].append(row['Response ID'])
            if len(row['Issues']) == 0 and 'No Label' in issues:
                issue_response_id_map[date]['No Label'].append(row['Response ID'])
    # Fill in component and issue df with 0 for Nan (?)
    issue_df = issue_df.fillna(0).astype(int).rename_axis('Issues')
    issue_df.drop([0], axis = 1, inplace = True)
    return issue_df, issue_response_id_map, issue_day_response_id_map


def updateGraph(df, title, num_days_range = 7):
    filtered_df = df.iloc[:, 0:num_days_range]
    traces = []
    # Checking df for values:
    for index, row in filtered_df.iterrows():
        print(list(row.keys()))
        traces.append(go.Bar(
            x=list(row.keys()),
            y=row.values,
            name=index,
            customdata=[index] * len(list(row.keys())),
            # hoverinfo='none',
            # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
            # customdata=docs.iloc[0].values
        ))
    # Stacked Bar Graph figure - components:
    layout = go.Layout(
        barmode='stack',
        title=title,
        font=dict(
            family='Montserrat, Helvetica Neue, Helvetica, sans-serif',
            size=12, 
            color='white'
        ),
        xaxis=dict(
            # showticklabels=False,
            title='Time'
        ),
        yaxis=dict(
            title='Count of Docs'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig = dict(data=traces, layout=layout)
    return fig


# CREATE FIRST TWO GRAPHS
day_range = min(results2_df['Day Difference'].max(), toggle_time_params['max'])
print(day_range)
component_df, comp_response_id_map, comp_day_response_id_map = initCompDF(results2_df, day_range)
list_component_df = component_df
issue_df, issue_response_id_map, issue_day_response_id_map = initIssueDF(results2_df, day_range)
list_issue_df = issue_df
fig_component = updateGraph(component_df, 'Components Over Time', 7)
fig_issue = updateGraph(issue_df, 'Issues Over Time', 7)


# DRILLDOWN FUNCTIONS
def drilldownClustering(df):
    results = runDrilldown(df)
    results = results.transpose()
    fig = clusteringBarGraph(results, 'Clustering Analysis')
    return fig


def clusteringBarGraph(df, title):
    traces = []
    # Get Count, Words, Phrases
    count = list(df.loc['Count'].values)
    ids = list(df.loc['Response IDs'].values)
    words = list(df.loc['Words'].values)
    phrases = list(df.loc['Phrases'].values)
    traces = [go.Bar(
            x=words,
            y=count,
            text = phrases,
            customdata=ids,
            hoverinfo='text',
        )]
    layout = go.Layout( #TODO: check
        title=title,
        font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
        xaxis=dict(
            # showticklabels=False,
            title='Time'
        ),
        yaxis=dict(
            title='Count of Docs'
        )
    )
    fig = dict(data=traces, layout=layout)
    return fig


#prep data for displaying in stacked binary sentiment graph over time
#Grab unique dates from results_df
results_df["Date Submitted"] = pd.to_datetime(results_df["Date Submitted"])
results_df['Day Difference'] = (reference - pd.to_datetime(results_df['Date Submitted'], format='%Y-%m-%d %H:%M:%S')).dt.days + 1

global_filtered_top_sites_df = results_df
global_filtered_other_sites_df = results_df

unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()


#compacted list of all sites mentioned in the comments
results_df['Sites'] = results_df['Sites'].apply(lambda s: s.replace("https://", "").replace("http://", "").replace("www.", ""))
results_df['Sites'] = results_df['Sites'].str.strip()
results_df['Sites'] = [x.strip() for x in results_df['Sites']]
results_df['Sites'] = ['facebook.com' if x == 'facebook' or x == ' facebook' else x for x in results_df['Sites']]
results_df['Sites'] = ['youtube.com' if x == 'youtube' or x == ' youtube' else x for x in results_df['Sites']]
results_df['Sites'] = ['amazon.com' if x == 'amazon' or x == ' amazon' else x for x in results_df['Sites']]
results_df['Sites'] = ['google.com' if x == 'google' or x == ' google' else x for x in results_df['Sites']]

sites_list = results_df['Sites'].apply(pd.Series).stack().reset_index(drop=True)
sites_list = ','.join(sites_list).split(',')
sites_list = [x for x in sites_list if '.' in x]

sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
sites_df = sites_df.sort_values(by=['Count'])
sites_df = sites_df[sites_df['Site'] != 'None Found']

top_sites_list = [item for item in sites_list if item in top_sites]
top_sites_df = pd.DataFrame.from_dict(Counter(top_sites_list), orient='index').reset_index()
top_sites_df = top_sites_df.rename(columns={'index': 'Site', 0: 'Count'})
top_sites_df = top_sites_df.sort_values(by=['Count'], ascending=False)


other_sites_list = [item for item in sites_list if item not in top_sites and item != 'None Found']
other_sites_df = pd.DataFrame.from_dict(Counter(other_sites_list), orient='index').reset_index()
other_sites_df = other_sites_df.rename(columns={'index': 'Site', 0: 'Count'})
other_sites_df = other_sites_df[other_sites_df['Count'] > 1]
other_sites_df = other_sites_df.sort_values(by=['Count'], ascending=False)

init_count = len(sites_df.index)


# Page styling - sample:

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)
# suppress exception of assigning callbacks to components that are genererated
# by other callbacks
app.title = 'Mozilla Analytics'
app.config.suppress_callback_exceptions = True
'''
Dash apps are composed of 2 parts. 1st part describes the app layout.
The 2nd part describes the interactivty of the app 
'''

################################################################################
###
### LAYOUT ###
###
################################################################################

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])



# prep data for displaying in stacked binary sentiment graph over time
# Grab unique dates from results_df
results_df["Date Submitted"] = pd.to_datetime(results_df["Date Submitted"])
unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()
common_df = test2 = results_df.groupby('Sites')['Sites'].agg(['count']).reset_index()
common_df = common_df.sort_values(by=['count'])


list_page_children = []

################################################################################
###
### CALLBACKS ###
###
################################################################################

from callbacks import *


################################################################################
###
### RUN ###
###
################################################################################

if __name__ == '__main__':
    app.run_server(debug=False)

