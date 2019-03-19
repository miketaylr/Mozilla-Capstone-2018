import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table_experiments as dte
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import ast
import json
from clustering import runDrilldown
from datetime import datetime as datetime
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE
from collections import Counter
import numpy as np
import urllib.parse
import os
import ast
import re
import dash_dangerously_set_inner_html


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',
                        'https://fonts.googleapis.com/css?family=Montserrat:300,100']
external_scripts = ['https://code.jquery.com/jquery-3.2.1.min.js', 'https://d3js.org/d3.v4.min.js']


# Reading in data:
results_df = pd.read_csv("./data/output_pipeline.csv", encoding ="ISO-8859-1")
results2_df = pd.read_csv("./data/output_pipeline.csv", encoding="ISO-8859-1")
sr_df = pd.read_csv("./data/output_spam_removed.csv", encoding="ISO-8859-1")

top_sites = ['google.com',
'youtube.com',
'facebook.com',
'baidu.com',
'wikipedia.org',
'yahoo.com',
'qq.com',
'tmall.com',
'taobao.com',
'twitter.com',
'amazon.com',
'google.co.in',
'instagram.com',
'vk.com',
'sohu.com',
'live.com',
'jd.com',
'reddit.com',
'yandex.ru',
'weibo.com',
'sina.com.cn',
'google.co.jp',
'login.tmall.com',
'360.cn',
'blogspot.com',
'linkedin.com',
'google.com.hk',
'netflix.com',
'google.com.br',
'pornhub.com',
'pages.tmall.com',
'google.co.uk',
'csdn.net',
'yahoo.co.jp',
'twitch.tv',
'office.com',
'google.ru',
'microsoftonline.com',
'alipay.com',
'mail.ru',
'google.fr',
'google.de',
'microsoft.com',
'ebay.com',
'bing.com',
'msn.com',
'aliexpress.com',
'whatsapp.com',
'naver.com',
'google.com.mx',
'xvideos.com',
'tribunnews.com',
'google.it',
'imdb.com',
'wordpress.com',
'stackoverflow.com',
'amazon.co.jp',
'google.es',
'google.ca',
'github.com',
'paypal.com',
'livejasmin.com',
'tumblr.com',
'google.com.tr',
'google.com.tw',
'imgur.com',
'google.co.kr',
'espn.com',
'xhamster.com',
'wikia.com',
'apple.com',
'pinterest.com',
'thestartmagazine.com',
'porn555.com',
'dropbox.com',
'xnxx.com',
'google.com.au',
'adobe.com',
'detail.tmall.com',
'cobalten.com',
'amazon.in',
'amazon.co.uk',
'hao123.com',
'amazon.de',
'quora.com',
'txxx.com',
'google.co.id',
'tianya.cn',
'bilibili.com',
'booking.com',
'google.co.th',
'xinhuanet.com',
'aparat.com',
'pixnet.net',
'salesforce.com',
'google.pl',
'rakuten.co.jp',
'bongacams.com',
'cnn.com',
'google.com.ar']

# print(results_df.shape) # SHOULD FILL NAN VALS AS WELL WHEN POSSIBLE
# search_df = results_df[["Response ID", "Date Submitted", "Country","City"\
#                         , "State/Region", "Binary Sentiment", "Positive Feedback"\
#                         , "Negative Feedback", "Relevant Site", "compound"\
#                         , "Sites", "Issues", "Components"]]
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
reference = datetime(2016, 12, 30)
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
            title = 'This Week in Overall Global Sentiment of Mozilla Web Compat',
            titlefont=dict(
                family='Helvetica Neue, Helvetica, sans-serif',
                color='#BCBCBC'
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
                family='Helvetica Neue, Helvetica, sans-serif',
                size=12,
                color='#BCBCBC',
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
# print(clusterNames)
df1 = df1.set_index('Issue')
docs = df1.drop(arrayOfNamesWords, axis=0)
words = df1.drop(arrayOfNamesDocs, axis=0)
# print(words.iloc[0].values[0])
clusters = df1.drop(['Words', 'Docs'], axis=0)
# print(clusters)


# Dynamic Data
# df2 = clustering.runVis(numClusters)
categoryDict = pd.Series(clusterDesc.description.values, index=clusterDesc.clusters_types).to_dict()
# docs = df2.tail(1)
# df2 = df2[:-1]
# phrases = df2.tail(1)
# df2 = df2[:-1]
# words = df2.tail(1)
# df2 = df2[:-1]
# clusters = df2
# clusters = clusters.rename(index=categoryDict)


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

# reference = datetime.now()

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
    component_df.drop(0, axis=1, inplace = True)
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
            family='Helvetica Neue, Helvetica, sans-serif', 
            size=12, 
            color='#BCBCBC'
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


# def mergedGraph():
#     # merge output_pipeline with output_clusters_defined
#     merged = pd.merge(results_df, clusters_df, on='Response ID')
#     merged = merged[merged['manual_clusters'].notna()]
#     return merged
# def updateCompMetricsGraph():
#     # CATEGORIZATION VISUALIZATION
#     merged = mergedGraph()
#     compCountSeries = pd.Series([])
#     # For components labelled:
#     for component in WORDS_TO_COMPONENT.keys():
#         compCounts = merged[merged['Components'].str.contains(component)]['manual_clusters'].value_counts()
#         compCountSeries = pd.concat([compCountSeries, compCounts.rename(component)], axis=1)
#     compCountSeries = pd.concat([compCountSeries, merged[merged['Components'].str.match("\[\]")][
#         'manual_clusters'].value_counts().rename('No Label')], axis=1)
#     compCountSeries = compCountSeries.drop(0, 1).fillna(0).astype(int)
#     compCountSeries = compCountSeries.rename(index=categoryDict)
#     traces_comp_metrics = []
#     for index, row in compCountSeries.iterrows():
#         # print(list(row.keys()))
#         traces_comp_metrics.append(go.Bar(
#             x=list(row.keys()),
#             y=row.values,
#             name=index,
#             # hoverinfo='none',
#             # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
#             # customdata=docs.iloc[0].values
#         ))
#     def update_point(trace):
#         # print(trace)
#         return
#     # Stacked Bar Graph figure - components labelled against manual labelling:
#     layout_comp_metrics = go.Layout(
#         barmode='stack',
#         title='Components vs Manual Clusters',
#         font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
#         xaxis=dict(
#             # showticklabels=False,
#             title='Components'
#         ),
#         yaxis=dict(
#             title='Count of Docs'
#         )
#     )
#     fig_comp_metrics = dict(data=traces_comp_metrics, layout=layout_comp_metrics)
#     return fig_comp_metrics


# def updateIssuesMetricsGraph():
#     # ISSUES VISUALIZATION
#     merged = mergedGraph()
#     # For issues labelled:
#     issueCountSeries = pd.Series([])
#     for issue in WORDS_TO_ISSUE.keys():
#         issueCounts = merged[merged['Issues'].str.contains(issue)]['manual_clusters'].value_counts()
#         issueCountSeries = pd.concat([issueCountSeries, issueCounts.rename(issue)], axis=1)
#     issueCountSeries = pd.concat([issueCountSeries, merged[merged['Components'].str.match("\[\]")][
#         'manual_clusters'].value_counts().rename('No Label')], axis=1)
#     issueCountSeries = issueCountSeries.drop(0, 1).fillna(0).astype(int)
#     issueCountSeries = issueCountSeries.rename(index=categoryDict)
#     traces_issue_metrics = []
#     for index, row in issueCountSeries.iterrows():
#         # print(list(row.keys()))
#         traces_issue_metrics.append(go.Bar(
#             x=list(row.keys()),
#             y=row.values,
#             name=index,
#             # hoverinfo='none',
#             # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
#             # customdata=docs.iloc[0].values
#         ))
#     # Stacked Bar Graph figure - issues labelled against manual labelling:
#     layout_issue_metrics = go.Layout(
#         barmode='stack',
#         title='Issues vs Manual Clusters',
#         font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
#         xaxis=dict(
#             # showticklabels=False,
#             title='Issues'
#         ),
#         yaxis=dict(
#             title='Count of Docs'
#         )
#     )
#     fig_issue_metrics = dict(data=traces_issue_metrics, layout=layout_issue_metrics)
#     return fig_issue_metrics
# fig_comp_metrics = updateCompMetricsGraph()
# fig_issue_metrics = updateIssuesMetricsGraph()


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
    layout = go.Layout(
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
PAGE_SIZE = 40
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)
# suppress exception of assigning callbacks to components that are genererated
# by other callbacks
app.title = 'Mozilla Analytics'
'''
Dash apps are composed of 2 parts. 1st part describes the app layout.
The 2nd part describes the interactivty of the app 
'''
# tabs_styles = {
#     'height': '44px',
#     'width': '600px',
#     'display': 'inline-block'
# }
# tab_style = {
#     # 'borderBottom': '1px solid #d6d6d6',
#     'margin': '5px 0px 5px 0px',
#     'padding': '11px',
#     # 'backgroundColor': 'rgb(30,30,30)',
#     'backgroundColor': 'transparent',
#     'border': 'none',
# }
# # sites_tab_style = {
# #     # 'borderBottom': '1px solid #d6d6d6',
# #     'margin': '5px 0px 5px 0px',
# #     'padding': '11px 14px 11px 14px',
# #     'backgroundColor': 'transparent',
# #     'font-weight': 'bold',
# #     'border-style': 'solid',
# #     'border-width': '1px',
# # }
# tab_selected_style = {
#     # 'border': 'none',
#     'borderTop': 'none',
#     'borderRight': 'none',
#     'borderLeft': 'none',
#     'borderBottom': '1px solid #white !important',
#     'backgroundColor': 'transparent',
#     'color': 'white',
#     'padding': '11px'
# }

app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])
main_layout = html.Div(children=[
    html.Div(id="header",
             children=[
                 html.A(id='home-link', children = [
                     html.Div(
                         id="left-header",
                         children=[
                             html.Img(id='logo', src='../assets/Mozilla-Firefox-icon.png'),
                             html.H1(
                                 children='MOZILLA',
                                 id="title",
                             ),
                             html.H6(
                                 children='user feedback analytics',
                                 id="subtitle"
                             )
                         ],
                     ),
                 ], href='https://wiki.mozilla.org/Compatibility', target='_blank'),
                dcc.Tabs(id="tabs-styled-with-inline", value='sites', children=[
                    dcc.Tab(label='Sentiment', value='sentiment', className='tab_style', selected_className='tab_selected_style'),
                    dcc.Tab(label='World View', value='geoview', className='tab_style', selected_className='tab_selected_style'),
                    dcc.Tab(label='Components', value='components', className='tab_style', selected_className='tab_selected_style'),
                    dcc.Tab(label='Issues', value='issues', className='tab_style', selected_className='tab_selected_style'),
                    dcc.Tab(label='Sites', value='sites', className='tab_style', selected_className='tab_selected_style'),
                    dcc.Tab(id='search-tab', value='search', className='tab_style', selected_className='tab_selected_style'),
                ], className='tabs_styles'),
    ]),
    html.Div(id='tabs-content-inline') # , className='tab-content'
])


# prep data for displaying in stacked binary sentiment graph over time
# Grab unique dates from results_df
results_df["Date Submitted"] = pd.to_datetime(results_df["Date Submitted"])
unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()
common_df = test2 = results_df.groupby('Sites')['Sites'].agg(['count']).reset_index()
common_df = common_df.sort_values(by=['count'])


list_page_children = []


sites_layout = html.Div(className='sites-layout', children=[
    html.H3('Site-Specific Feedback', className='page-title'),
    html.Div([
        html.Label('Select Date Range:'),
        dcc.DatePickerRange(
            id='sites-date-range',
            min_date_allowed=results_df['Date Submitted'].min(),
            max_date_allowed=results_df['Date Submitted'].max(),
            start_date=results_df['Date Submitted'].min(),
            end_date=results_df['Date Submitted'].max()
        ),
        html.Div(id='unique-site-count'),
        # html.Div(id='slider-container', className='slider-container', children=[
        #     html.Div(dcc.Slider(
        #         id='sites-slider',
        #         min=0,
        #         max=init_count,
        #         value=init_count,
        #         marks={i: '{} sites'.format(i) for i in range(init_count) if (i % 30 == 0) or (i == init_count)}),
        #         style={'height': '50px', 'width': '100%', 'display': 'inline-block'}),
        #     html.Div(id='sites-slider-output')
        # ]),
    ]),
    # html.Div([
    #     dcc.Graph(
    #         id='top-mentioned-site-graph',
    #         figure={
    #             'data': [{
    #                 'x': top_sites_df['Count'],
    #                 'y': top_sites_df['Site'],
    #                 'orientation': 'h',
    #                 'customdata': top_sites_df['Site'],
    #                 'type': 'bar',
    #                 'marker': {
    #                     'color': '#E3D0FF'
    #                 },
    #             }],
    #             'layout': {
    #                 'title': "Feedback by Top Mentioned Site(s)",
    #                 'titlefont': {
    #                     'family': 'Helvetica Neue, Helvetica, sans-serif',
    #                     'color': '#BCBCBC',
    #                 },
    #                 'xaxis': {
    #                     'title': 'Number of Feedback'
    #                 },
    #                 'yaxis': {
    #                     'title': 'Website'
    #                 },
    #                 'font': {
    #                     'family': 'Helvetica Neue, Helvetica, sans-serif',
    #                     'size': 12,
    #                     'color': '#BCBCBC',
    #                 },
    #                 'paper_bgcolor': 'rgba(0,0,0,0)',
    #                 'plot_bgcolor': 'rgba(0,0,0,0)',
    #             },
    #         },
    #     ),
    # ]),
    # html.Div([
    #     dcc.Graph(
    #         id='other-mentioned-site-graph',
    #         figure={
    #             'data': [{
    #                 'x': other_sites_df['Count'],
    #                 'y': other_sites_df['Site'],
    #                 'orientation': 'h',
    #                 'customdata': other_sites_df['Site'],
    #                 'type': 'bar',
    #                 'marker': {
    #                     'color': '#E3D0FF'
    #                 },
    #             }],
    #             'layout': {
    #                 'title': "Feedback by Other Mentioned Site(s)",
    #                 'titlefont': {
    #                     'family': 'Helvetica Neue, Helvetica, sans-serif',
    #                     'color': '#BCBCBC',
    #                 },
    #                 'xaxis': {
    #                     'title': 'Number of Feedback'
    #                 },
    #                 'yaxis': {
    #                     'title': 'Website'
    #                 },
    #                 'font': {
    #                     'family': 'Helvetica Neue, Helvetica, sans-serif',
    #                     'size': 12,
    #                     'color': '#BCBCBC',
    #                 },
    #                 'paper_bgcolor': 'rgba(0,0,0,0)',
    #                 'plot_bgcolor': 'rgba(0,0,0,0)',
    #             },
    #         },
    #     ),
    # ]),
    html.Div(
        className='sites-table-container',
        children=[
            html.Div(id='top-sites-table-container', className='sites-table',
                children=[
                    html.H4('Alexa Top 100 Sites', className='page-title'),
                    # html.Div(id='top_sites_container', className='slider-container', children=[
                    #     html.Div(id='top_sites_slider_output'),
                    #     dcc.Slider(
                    #         id='top_sites_time_slider',
                    #         min=toggle_time_params['min'],
                    #         max=toggle_time_params['max'],
                    #         step=toggle_time_params['step'],
                    #         value=toggle_time_params['default'],
                    #         marks=toggle_time_params['marks']
                    #     ),
                    # ]),
                    html.Button("View Selected Data", id="top-view-selected", className="view-selected-data-disabled"),
                    dte.DataTable(  # Add fixed header row
                        id='top-sites-table',
                        rows=top_sites_df.to_dict('rows'),
                        row_selectable=True,
                        filterable=True,
                        sortable=True,
                        selected_row_indices=[],
                        editable=False,
                        max_rows_in_viewport=6,
                    ),
                ]
            ),
            html.Div(id='other-sites-table-container', className='sites-table',
                children=[
                    html.H4('Other Sites', className='page-title'),
                    # html.Div(id='other_sites_container', className='slider-container', children=[
                    #     html.Div(id='other_sites_slider_output'),
                    #     dcc.Slider(
                    #         id='other_sites_time_slider',
                    #         min=toggle_time_params['min'],
                    #         max=toggle_time_params['max'],
                    #         step=toggle_time_params['step'],
                    #         value=toggle_time_params['default'],
                    #         marks=toggle_time_params['marks']
                    #     ),
                    # ]),
                    html.Button("View Selected Data", id="other-view-selected", className="view-selected-data-disabled"),
                    dte.DataTable(  # Add fixed header row
                        id='other-sites-table',
                        rows=other_sites_df.to_dict('rows'),
                        row_selectable=True,
                        filterable=True,
                        sortable=True,
                        selected_row_indices=[],
                        editable=False,
                        max_rows_in_viewport=6,
                    ),
                ]
            ),
    ]),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Div(className="close-button-container", children=[
                html.Button("Close", id="top-close-modal-site", className="close", n_clicks=0),
            ]),
            html.H2("Selected Feedback Data Points", className='modal-title'),  # Header
            html.Div(className='drill-down-container', children=[
                html.A("Analytics", className='drill-down-link', href='/sites-classification', target="_blank"), # close button
                html.A("Clustering", className='drill-down-link', href='/sites-clustering', target="_blank"),
                html.A("Download CSV", id='top-download-sites-link', className='download-link', href='', target="_blank"),
            ]),
            html.Div(className='top-modal-table-container', children=[
                dte.DataTable(  # Add fixed header row
                    id='top-modal-site-table',
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                    column_widths=['30%', '10%', '10%', '10%', '10%', '10%'],
                ),
            ]),
        ], id='top-modal-content-site', className='modal-content')
    ], id='top-modal-site', className='modal'),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Div(className="close-button-container", children=[
                html.Button("Close", id="other-close-modal-site", className="close", n_clicks=0),
            ]),
            html.H2("Selected Feedback Data Points", className='modal-title'),  # Header
            html.Div(className='drill-down-container', children=[
                html.A("Analytics", className='drill-down-link', href='/sites-classification', target="_blank"), # close button
                html.A("Clustering", className='drill-down-link', href='/sites-clustering', target="_blank"),
                html.A("Download CSV", id='other-download-sites-link', className='download-link', href='', target="_blank"),
            ]),
            html.Div(className='modal-table-container', children=[
                dte.DataTable(  # Add fixed header row
                    id='other-modal-site-table',
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                ),
            ]),
        ], id='other-modal-content-site', className='modal-content')
    ], id='other-modal-site', className='modal'),
])


sentiment_layout = html.Div([
    html.H3('How Do Users Feel About Mozilla?', className='page-title'),
    html.Div([
        html.Div(
            children=[
            dcc.RadioItems(
                id='sentiment-frequency',
                options=[
                    {'label': 'Daily', 'value': 'D'},
                    {'label': 'Weekly', 'value': 'W'},
                    {'label': 'Monthly', 'value': 'M'}
                ],
                value='D',
                labelStyle={'display': 'inline-block'},
                inputStyle = {'background-color': 'orange'}
            ),
            dcc.Graph(
               id='binary-sentiment-ts',
               figure={
                    'data': [
                        {
                            'x': unique_dates,
                            'y': results_df[results_df["Binary Sentiment"] == "Sad"].groupby(
                                [results_df['Date Submitted'].dt.date])['Binary Sentiment'].count().values,
                            'type': 'scatter',
                            'name': "Sad"
                        },
                        {
                            'x': unique_dates,
                            'y': results_df[results_df["Binary Sentiment"] == "Happy"].groupby([results_df['Date Submitted'].dt.date])['Binary Sentiment'].count().values,
                            'type': 'scatter',
                            'name': "Happy"
                        }
                    ],
                    'layout': {
                        'title': "Happy/Sad Sentiment Breakdown",
                        'titlefont': {
                            'family': 'Helvetica Neue, Helvetica, sans-serif',
                            'color': '#BCBCBC',
                        },
                        'xaxis': {
                            'title': 'Time'
                        },
                        'yaxis': {
                            'title': 'Amount of Feedback'
                        },
                        'font': {
                            'family': 'Helvetica Neue, Helvetica, sans-serif',
                            'size': 12,
                            'color': '#BCBCBC',
                        },
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)'
                    }
                }
            )]
        ),
    ]),
    html.Div([
        html.Div([ #entire modal
                #modal content
            html.Div([
                html.Button("Close", id="close-modal", className="close", n_clicks_timestamp=0), #close button
                html.H2("Selected Feedback Data Points"),#Header
                dt.DataTable(
                    id='modal-table',
                    columns=[{"name": i, "id": i} for i in search_df.columns],
                    pagination_settings={
                        'current_page': 0,
                        'page_size': PAGE_SIZE
                    },
                    pagination_mode='be',
                    sorting='be',
                    sorting_type='single',
                    sorting_settings=[],
                    n_fixed_rows=1,
                    style_table={
                     'overflowX': 'scroll',
                     'maxHeight': '800',
                     'overflowY': 'scroll'
                    },
                    style_cell={
                        'minWidth': '50'
                                 'px', 'maxWidth': '200px',
                        'whiteSpace': 'no-wrap',
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                    },
                    style_cell_conditional=[{
                        'if': {'column_id': 'Feedback'},
                        'textAlign': 'left'
                    }],
                    css=[{
                     'selector': '.dash-cell div.dash-cell-value',
                     'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',

                    }],
                )
             ], id='modal-content', className='modal-content')
         ], id='modal', className='modal'),
        html.Div(
            className='six columns',
            id='current-content'
        )
    ])
])


geoview_layout = html.Div(className='geo-layout', children = [
    html.H3('How Does The World Feel About Mozilla?', className='page-title'),
    html.Div(id='geo_container', className='slider-container', children=[
        html.Div(id='geo_slider_output'),
        dcc.Slider(
            id='geo_time_slider',
            min=toggle_time_params['min'],
            max=toggle_time_params['max'],
            step=toggle_time_params['step'],
            value=toggle_time_params['default'],
            marks=toggle_time_params['marks']
        ),
    ]),
    dcc.RadioItems(
        id='geoview-radio',
        style={'text-align': 'center'},
        options=[
            {'label': 'For the Time Range Selected Above', 'value': 'week'},
            {'label': 'Normalized by Country', 'value': 'norm'},
            {'label': 'Normalized Globally', 'value': 'globalNorm'},
        ],
        value='week'
    ),
    dcc.Graph(
        id='country-graph',
        figure=fig_geo),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Div(className="close-button-container", children=[
                html.Button("Close", id="close-geo-modal", className="close", n_clicks=0),
            ]),
            html.H2("Selected Feedback Data Points", className='modal-title'),  # Header
            html.Div(className='drill-down-container', children=[
                html.A("Analytics", className='drill-down-link', href='/geo-classification', target="_blank"), # close button
                html.A("Clustering", className='drill-down-link', href='/geo-clustering', target="_blank"),
                html.A("Download CSV", id='download-geo-link', className='download-link', href='', target="_blank"),
            ]),
            html.Div(className='modal-table-container', children=[
                dte.DataTable(  # Add fixed header row
                    id='modal-geo-table',
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                ),
            ]),
        ], id='modal-content-geo', className='modal-content')
    ], id='modal-geo', className='modal'),
])


components_layout = html.Div(className='sites-layout', children=[
    html.H3('Where Did Mozilla Break?', className='page-title'),
    html.Div(id='comp_container', className='slider-container', children=[
        html.Div(id='comp_slider_output'),
        dcc.Slider(
            id='comp_time_slider',
            min=toggle_time_params['min'],
            max=toggle_time_params['max'],
            step=toggle_time_params['step'],
            value=toggle_time_params['default'],
            marks=toggle_time_params['marks']
        ),
    ]),
    html.Div([
        html.Div(children=[
            dcc.Graph(id='comp-graph', figure=fig_component),
        ]),
    ]),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Div(className="close-button-container", children=[
                html.Button("Close", id="close-comp-site", className="close", n_clicks=0),
            ]),
            html.H2("Selected Feedback Data Points", className='modal-title'),  # Header
            html.Div(className='drill-down-container', children=[
                html.A("Analytics", className='drill-down-link', href='/comp-classification', target="_blank"), # close button
                html.A("Clustering", className='drill-down-link', href='/comp-clustering', target="_blank"),
                html.A("Download CSV", id='download-comp-link', className='download-link', href='', target="_blank"),
            ]),
            html.Div(className='modal-table-container', children=[
                dte.DataTable(  # Add fixed header row
                    id='modal-comp-table',
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                ),
            ]),
        ], id='modal-content-comp', className='modal-content')
    ], id='modal-comp', className='modal'),
])


issues_layout = html.Div(className='sites-layout', children=[
    html.H3('Browser Issues', className='page-title'),
    html.Div(id='issue_container', className='slider-container', children=[
        html.Div(id='issue_slider_output'),
        dcc.Slider(
            id='issue_time_slider',
            min=toggle_time_params['min'],
            max=toggle_time_params['max'],
            step=toggle_time_params['step'],
            value=toggle_time_params['default'],
            marks=toggle_time_params['marks']
        ),
    ]),
    html.Div([
        html.Div(children=[
            dcc.Graph(id='issue-graph', figure=fig_issue),
        ])
    ]),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Div(className="close-button-container", children=[
                html.Button("Close", id="close-issue-site", className="close", n_clicks=0),
            ]),
            html.H2("Selected Feedback Data Points", className='modal-title'),  # Header
            html.Div(className='drill-down-container', children=[
                html.A("Analytics", className='drill-down-link', href='/issues-classification', target="_blank"), # close button
                html.A("Clustering", className='drill-down-link', href='/issues-clustering', target="_blank"),
                html.A("Download CSV", id='download-issues-link', className='download-link', href='', target="_blank"),
            ]),
            html.Div(className='modal-table-container', children=[
                dte.DataTable(  # Add fixed header row
                    id='modal-issue-table',
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                ),
            ]),
        ], id='modal-content-issue', className='modal-content')
    ], id='modal-issue', className='modal'),
])


search_layout = html.Div([
    html.Img(src='https://loading.io/assets/img/loader/msg.gif', id = 'search-loading', style={'display': 'none'}),
    html.H3('Search Feedback', className='page-title'),
    # html.Label('Enter Search Request:'),
    dcc.Input(id='searchrequest', type='text', placeholder='Type Here'),
    html.Div(id='search-count-reveal'),
    html.A("Download CSV", id='search-download-link', className='download-link search-download-link', href='', target="_blank", style={'display': 'none'}),
    html.Div(id='search-table-container',
             children=[
                 dte.DataTable(  # Add fixed header row
                     id='searchtable',
                     rows=[{}],
                     row_selectable=False,
                     filterable=True,
                     sortable=True,
                     selected_row_indices=[],
                 ),
             ],
             style={'visibility': 'hidden'})
    ],
    style={'text-align': 'center'})


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    # this is where you put the stuff
    # set the data you need into a variable inside the click callback on the graph (the same place where you open the modal)
    # reference that variable here to build the page contents
    global global_site_modal_ids
    global global_comp_issue_modal_ids
    global list_component_df
    global list_issue_df
    global global_top_selected_sites

    if pathname is None:
        return main_layout

    ids = []
    if 'sites' in pathname:
        ids = global_site_modal_ids
    elif 'comp' in pathname:
        ids = global_comp_modal_ids
    elif 'issues' in pathname:
        ids = global_issue_modal_ids
    elif 'geo' in pathname:
        ids = global_geo_modal_ids

    if 'classification' in pathname:
        # global_sites_list_df is the dataframe that contains the data that appears in the modal
        # ideally this should be fed through the same functions as results2_df to create the figures to display on the new page
        results_modal_df = results2_df[results2_df['Response ID'].isin(ids)]
        day_range_site_list = min(results_modal_df['Day Difference'].max(), toggle_time_params['max'])

        component_df_list, comp_response_id_map_list, comp_day_response_id_map_list = initCompDF(results_modal_df, day_range_site_list)
        list_component_df = component_df_list
        issue_df_list, issue_response_id_map_list, issue_day_response_id_map_list = initIssueDF(results_modal_df, day_range_site_list)
        list_issue_df = issue_df_list
        fig_component_list = updateGraph(component_df_list, 'Components Over Time', 7)
        fig_issue_list = updateGraph(issue_df_list, 'Issues Over Time', 7)
        return html.Div([
            html.Div([
                html.H1(
                    children='Mozilla Web Compatability Analytics',
                    id="title",
                ),
            ]),
            html.Div([
                html.Div(id='list_comp_container',
                         className='one-half column list-slider-container',
                         children=[
                            html.H3('Components', className='page-title'),
                            html.Div(id='list_comp_slider_output'),
                            dcc.Slider(id='list_comp_time_slider',
                                        className='list-page-slider',
                                        min=toggle_time_params['min'], max=toggle_time_params['max'],
                                        step=toggle_time_params['step'], value=toggle_time_params['default'],
                                        marks=toggle_time_params['marks']),
                            dcc.Graph(id='list-comp-graph', figure=fig_component_list),
                         ]
                 ),
                html.Div(id='list_issue_container',
                         className='one-half column list-slider-container',
                         children=[
                            html.H3('Issues', className='page-title'),
                            html.Div(id='list_issue_slider_output'),
                            dcc.Slider(id='list_issue_time_slider',
                                        className='list-page-slider',
                                        min=toggle_time_params['min'], max=toggle_time_params['max'],
                                        step=toggle_time_params['step'], value=toggle_time_params['default'],
                                        marks=toggle_time_params['marks']),
                            dcc.Graph(id='list-issue-graph', figure=fig_issue_list),
                ]),
                ]),
            # html.Ul([html.Li(x) for x in global_top_selected_sites])
        ])
    elif 'clustering' in pathname:
        # global_sites_list_df is the dataframe that contains the data that appears in the modal
        # ideally this should be fed through the same functions as results2_df to create the figures to display on the new page
        results_modal_df = sr_df[sr_df['Response ID'].isin(ids)]
        # day_range_site_list = min(results_modal_df['Day Difference'].max(), toggle_time_params['max'])
        results = runDrilldown(results_modal_df)
        results = results.sort_values(by='Count', ascending=False)
        results = results.reset_index()
        pageChildren = []
        rootDict = dict()
        rootDict['name'] = ''
        rootDict['children'] = []
        for index, cluster in results.iterrows():
            responseIds = cluster['Response IDs']
            listArray = []
            for response in responseIds:
                feedback = results2_df[results2_df['Response ID'] == response]
                listArray.append(html.Li(feedback['Feedback']))
            dataDict = dict()
            wordArr = []
            topWords = cluster['Words'].split(',')
            if topWords[0]:
                wordArr.append(topWords[0])
            if topWords[1]:
                wordArr.append(topWords[1])
            dataDict['name'] = ",".join(wordArr)
            dataDict['value'] = len(listArray)
            clusterId = 'cluster_' + str(index+1)
            dataDict['id'] = clusterId
            rootDict['children'].append(dataDict)
            child=html.Details([
                html.Summary(('Cluster ' + str(index+1) + ' - ' + cluster['Words']), id=clusterId, className='clustering-summary-text'),
                html.P('Top Phrases: ' + cluster['Phrases'], className='clustering-top-text'),
                html.P('Feedback: ', className='clustering-top-text'),
                html.Ul(children=listArray, className='clustering-feedback'),
            ])
            # child = html.Div(className='clustering-group', children=[
            #     html.H3('Cluster ' + str(index+1)),
            #     html.P('Top Words: ' + cluster['Words'], className='clustering-top-text'),
            #     html.P('Top Phrases: ' + cluster['Phrases'], className='clustering-top-text'),
            #     html.P('Feedback: ', className='clustering-top-text'),
            #     html.Ul(children=listArray, className='clustering-feedback'),
            # ])
            pageChildren.append(child)

        results = results.transpose()
        fig = clusteringBarGraph(results, 'Clustering Analysis')
        return html.Div(className='clustering-page', children=[
            html.Div([
                html.H1(
                    children='Mozilla Web Compat Analytics',
                    id="title",
                ),
            ]),
            html.Div(id='div-d3', className="d3-container", **{'data-d3': json.dumps(rootDict)}, children=[
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
                <svg width="800" height="800"></svg>
            '''),
            ]),
            html.Div(children=pageChildren),
            # html.Div([
            #     html.Div(id='list_comp_container',
            #              className='list-slider-container',
            #              children=[
            #                 html.H3('Clustered Data', className='page-title'),
            #                  dcc.Graph(id='cluster-graph', figure=fig),
            #              ]
            #      ),
            #     ]),
            # html.Div(id='cluster-table-container',
            #          children=[
            #              dte.DataTable(  # Add fixed header row
            #                  id='cluster-table',
            #                  rows=[{}],
            #                  row_selectable=True,
            #                  filterable=True,
            #                  sortable=True,
            #                  selected_row_indices=[],
            #              ),
            #          ]),
            # html.Ul([html.Li(x) for x in global_top_selected_sites])
        ])
    else:
        return main_layout


# @app.callback(Output('url', 'pathname'),
#               [Input('tabs-styled-with-inline', 'value')])
# def update_url(tab):  # bit of a hacky way of updating URL for now.
#     print("clicked tab", tab)
#     return tab

@app.callback(Output('cluster-table', 'rows'),
              [Input('cluster-graph', 'clickData')])
def update_table(clickData):
    ids = clickData['points'][0]['customdata']
    dff = search_df[search_df['Response ID'].isin(ids)]
    cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
              'Feedback', 'Components', 'Issues', 'Sites']
    cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                 'Feedback', 'Components', 'Issues', 'Sites']
    dff = dff[cnames]
    dff.columns = cnamesnew
    return dff.to_dict('rows')


@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])
def render_content(tab):
    if tab == 'sites':
        return sites_layout
    elif tab == 'sentiment':
        return sentiment_layout
    elif tab == 'geoview':
        return geoview_layout
    elif tab == 'components':
        return components_layout
    elif tab == 'issues':
        return issues_layout
    elif tab == 'search':
        return search_layout
    else:
        return sites_layout

@app.callback(Output('binary-sentiment-ts', 'figure'),
              [Input('sentiment-frequency', 'value')])
def update_sentiment_graph(frequency):

    sad_df = (
        results_df[results_df["Binary Sentiment"] == "Sad"].groupby([pd.Grouper(key="Date Submitted", freq=frequency)]).count()
        .reset_index()
        .sort_values("Date Submitted")
    )

    happy_df = (
        results_df[results_df["Binary Sentiment"] == "Happy"].groupby([pd.Grouper(key="Date Submitted", freq=frequency)]).count()
        .reset_index()
        .sort_values("Date Submitted")
    )

    fig = {
        'data': [
            {
                'x': sad_df['Date Submitted'],
                'y': sad_df['Binary Sentiment'],
                'type': 'scatter',
                'name': "Sad"
            },
            {
                'x': happy_df['Date Submitted'],
                'y': happy_df['Binary Sentiment'],
                'type': 'scatter',
                'name': "Happy"
            }
        ],
        'layout': {
            'title': "Happy/Sad Breakdown",
            'titlefont': {
                'family': 'Helvetica Neue, Helvetica, sans-serif',
                'color': '#BCBCBC',
            },
            'xaxis': {
                'title': 'Time'
            },
            'yaxis': {
                'title': 'Amount of Feedback'
            },
            'font': {
                'family': 'Helvetica Neue, Helvetica, sans-serif',
                'size': 12,
                'color': '#BCBCBC',
            },
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)'#,
            #'barmode': 'stack',
        }
    }


    return fig



# @app.callback(
#     Output('modal-sentiment', 'style'),
#     [Input('close-sentiment-modal', 'n_clicks'),
#      Input('binary-sentiment-ts', 'clickData')])
# def display_senti_modal(closeClicks, clickData):
#     global sentiCloseCount
#     print('here in sentiment', closeClicks, sentiCloseCount)  # clickData exists, but running into an exception here....
#     if closeClicks > sentiCloseCount:
#         sentiCloseCount = closeClicks
#         return {'display': 'none'}
#     elif clickData:
#         return {'display': 'block'}
#     else:
#         return {'display': 'none'}
#
#
# @app.callback(
#     Output('modal-sentiment-table', 'rows'),
#     [Input('binary-sentiment-ts', 'clickData')])
# def display_senti_click_data(clickData):
#     #Set click data to whichever was clicked
#     print('here in sentiment', clickData)
#     if (clickData):
#         global global_senti_modal_ids
#         if(len(clickData['points']) == 1):
#             day = clickData['points'][0]['x']
#             component = clickData['points'][0]['customdata']
#             global senti_response_id_map
#             ids = senti_response_id_map[day][component]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#         else:
#             day = clickData['points'][0]['x']
#             global senti_day_response_id_map
#             ids = senti_day_response_id_map[day]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#
#         cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         dff = dff[cnames]
#         dff.columns = cnamesnew
#         global_senti_modal_ids = ids
#         return dff.to_dict('rows')
#     else:
#         return []
#
#
# @app.callback(
#     Output('download-senti-link', 'href'),
#     [Input('binary-sentiment-ts', 'clickData')])
# def update_senti_download_link(clickData):
#     if clickData:
#         if(len(clickData['points']) == 1):
#             day = clickData['points'][0]['x']
#             comp = clickData['points'][0]['customdata']
#             global senti_response_id_map
#             ids = senti_response_id_map[day][comp]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#         else:
#             day = clickData['points'][0]['x']
#             global senti_day_response_id_map
#             ids = senti_day_response_id_map[day]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#
#         cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         dff = dff[cnames]
#         csv_string = dff.to_csv(index=False, encoding='utf-8')
#         csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
#         return csv_string
#     else:
#         return ''
#
#
# @app.callback(
#     Output('modal-geo', 'style'),
#     [Input('close-geo-modal', 'n_clicks'),
#      Input('country-graph', 'clickData')])
# def display_senti_modal(closeClicks, clickData):
#     global geoCloseCount
#     print('here in geo', closeClicks, geoCloseCount)
#     if closeClicks > geoCloseCount:  # running into exception here...
#         geoCloseCount = closeClicks
#         return {'display': 'none'}
#     elif clickData:
#         return {'display': 'block'}
#     else:
#         return {'display': 'none'}
#
#
# @app.callback(
#     Output('modal-geo-table', 'rows'),
#     [Input('country-graph', 'clickData')])
# def display_senti_click_data(clickData):
#     #Set click data to whichever was clicked
#     print('here in geo', clickData)
#     if (clickData):
#         global global_geo_modal_ids
#         if(len(clickData['points']) == 1):
#             day = clickData['points'][0]['x']
#             component = clickData['points'][0]['customdata']
#             global geo_response_id_map
#             ids = geo_response_id_map[day][component]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#         else:
#             day = clickData['points'][0]['x']
#             global geo_day_response_id_map
#             ids = geo_day_response_id_map[day]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#         cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         dff = dff[cnames]
#         dff.columns = cnamesnew
#         global_geo_modal_ids = ids
#         return dff.to_dict('rows')
#     else:
#         return []
#
#
# @app.callback(
#     Output('download-geo-link', 'href'),
#     [Input('country-graph', 'clickData')])
# def update_senti_download_link(clickData):
#     if clickData:
#         if(len(clickData['points']) == 1):
#             day = clickData['points'][0]['x']
#             comp = clickData['points'][0]['customdata']
#             global geo_response_id_map
#             ids = geo_response_id_map[day][comp]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#         else:
#             day = clickData['points'][0]['x']
#             global geo_day_response_id_map
#             ids = geo_day_response_id_map[day]
#             dff = search_df[search_df['Response ID'].isin(ids)]
#
#         cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         dff = dff[cnames]
#         csv_string = dff.to_csv(index=False, encoding='utf-8')
#         csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
#         return csv_string
#     else:
#         return ''


@app.callback(
    Output('modal-comp', 'style'),
    [Input('close-comp-site', 'n_clicks'),
     Input('comp-graph', 'clickData')])
def display_comp_modal(closeClicks, clickData):
    global compCloseCount
    print('here in comp', closeClicks, compCloseCount)
    if closeClicks > compCloseCount:
        compCloseCount = closeClicks
        return {'display': 'none'}
    elif clickData:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('modal-comp-table', 'rows'),
    [Input('comp-graph', 'clickData')])
def display_comp_click_data(clickData):
    #Set click data to whichever was clicked
    print('here in comp', clickData)
    if (clickData):
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            component = clickData['points'][0]['customdata']
            global comp_response_id_map
            ids = comp_response_id_map[day][component]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            global comp_day_response_id_map
            ids = comp_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global global_comp_modal_ids
        global_comp_modal_ids = list(dff['Response ID'])
        return dff.to_dict('rows')
    else:
        return []


@app.callback(
    Output('download-comp-link', 'href'),
    [Input('comp-graph', 'clickData')])
def update_comp_download_link(clickData):
    if clickData:
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            comp = clickData['points'][0]['customdata']
            global comp_response_id_map
            ids = comp_response_id_map[day][comp]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            global comp_day_response_id_map
            ids = comp_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ''


@app.callback(
    Output('modal-issue', 'style'),
    [Input('close-issue-site', 'n_clicks'),
     Input('issue-graph', 'clickData')])
def display_issue_modal(closeClicks, clickData):
    global issueCloseCount
    print('here in issue', closeClicks, issueCloseCount)
    if closeClicks > issueCloseCount:
        issueCloseCount = closeClicks
        return {'display': 'none'}
    elif clickData:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('modal-issue-table', 'rows'),
    [Input('issue-graph', 'clickData')])
def display_issue_click_data(clickData):
    #Set click data to whichever was clicked
    print('here in issue', clickData)
    if (clickData):
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            issue = clickData['points'][0]['customdata']
            global issue_response_id_map
            ids = issue_response_id_map[day][issue]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            global issue_day_response_id_map
            ids = issue_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global global_issue_modal_ids
        global_issue_modal_ids = list(dff['Response ID'])
        return dff.to_dict('rows')
    else:
        return []


@app.callback(
    Output('download-issues-link', 'href'),
    [Input('issue-graph', 'clickData')])
def update_issue_download_link(clickData):
    if(clickData):
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            issue = clickData['points'][0]['customdata']
            global issue_response_id_map
            ids = issue_response_id_map[day][issue]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            global issue_day_response_id_map
            ids = issue_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ''


# @app.callback(Output('modal', 'style'), [Input('display_data','n_clicks_timestamp'),
#                                          Input('close-modal', 'n_clicks_timestamp')])
# def display_modal(openm, closem):
#     if closem > openm:
#         return {'display': 'none'}
#     elif openm > closem:
#         return {'display': 'block'}
#     else:
#         return {}
# @app.callback(
#     Output('modal-table', "data"),
#     [Input('modal-table', "pagination_settings"),
#      Input('modal-table', "sorting_settings"),
#      Input('display_data','n_clicks_timestamp'),
#      Input('close-modal', 'n_clicks_timestamp'),
#      Input('trends-scatterplot', 'selectedData')])
# def update_modal_table(pagination_settings, sorting_settings, openm, closem, selectedData):
#     if openm > closem: # only update the table if the modal is open
#         ids = list(d['customdata'] for d in selectedData['points'])
#         dff = search_df[search_df['Response ID'].isin(ids)]

#         if len(sorting_settings):
#             dff = dff.sort_values(
#                 [col['column_id'] for col in sorting_settings],
#                 ascending=[
#                     col['direction'] == 'asc'
#                     for col in sorting_settings
#                 ],
#                 inplace=False
#             )

#         return dff.iloc[
#                pagination_settings['current_page'] * pagination_settings['page_size']:
#                (pagination_settings['current_page'] + 1) * pagination_settings['page_size']
#                ].to_dict('rows')
#     else:
#         return {}
# @app.callback(
#     Output('current-content', 'children'),
#     [Input('trends-scatterplot', 'hoverData')])
# def display_hover_data(hoverData):
#     # try: # Get the row from the results
#     #     r = results_df[results_df['Response ID'] == hoverData['points'][0]['customdata']]
#     #     return html.H4(
#     #         "The comment from {} is '{}'. The user was {}.".format(
#     #             r.iloc[0]['Date Submitted'],
#     #             r.iloc[0]['Feedback'],
#     #             r.iloc[0]['Binary Sentiment']
#     #         )
#     #     )
#     # except TypeError:
#     #     print('no hover data selected yet')
#     # return ''
#     return
# @app.callback(
#     Output('trend-data-histogram', 'figure'),
#     [Input('trends-scatterplot', 'selectedData')])
# def display_selected_trend_data(selectedData):
#     # return table matching the current selection
#     ids = list(d['customdata'] for d in selectedData['points'])
#     df = search_df[search_df['Response ID'].isin(ids)]
#     # print(ids)
#     return {
#         'data': [
#             {
#                 'x': df['compound'],
#                 'name': 'Compound Sentiment',
#                 'type': 'histogram',
#                 'autobinx': True
#             }
#         ],
#         'layout': {
#             'margin': {'l': 40, 'r': 20, 't': 0, 'b': 30}
#         }
#     }
@app.callback(
    Output('top-view-selected', 'className'),
    [Input('top-sites-table', 'selected_row_indices')])
def disable_site_modal_button(selected_row_indices):
    if selected_row_indices:
        return 'view-selected-data'
    else:
        return 'view-selected-data-disabled'

@app.callback(
    Output('top-modal-site-table', 'rows'),
    [Input('top-view-selected', 'n_clicks')],
    [State('top-sites-table', 'rows'),
     State('top-sites-table', 'selected_row_indices')])
def update_site_modal_table(clicks, rows, selected_row_indices):
    print('here', clicks)

    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    if(clicks):
        # ids = list(d['customdata'] for d in selectedData['points'])
        global global_filtered_top_sites_df
        dff = global_filtered_top_sites_df[global_filtered_top_sites_df['Sites'].isin(sites)]
        global global_site_modal_ids
        global_site_modal_ids = list(dff['Response ID'])

        cnames = ['Feedback', 'Date Submitted', 'Country',
                    'Components', 'Issues', 'Sites']
        cnamesnew = ['Feedback', 'Date Submitted', 'Country',
                   'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        print(global_site_modal_ids)
        global global_top_selected_sites
        global_top_selected_sites = sites
        return dff.to_dict('rows')
    return []

@app.callback(Output('top-modal-site', 'style'),
              [Input('top-close-modal-site', 'n_clicks'),
               Input('top-view-selected', 'n_clicks')],
              [State('top-sites-table', 'selected_row_indices')])
def display_modal(closeClicks, openClicks, selected_row_indices):
    global topSiteCloseCount
    if len(selected_row_indices) == 0:
        return {'display': 'none'}
    elif closeClicks > topSiteCloseCount:
        topSiteCloseCount = closeClicks
        return {'display': 'none'}
    elif openClicks:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    dash.dependencies.Output('top_sites_slider_output', 'children'),
    [dash.dependencies.Input('top_sites_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)

@app.callback(
    dash.dependencies.Output('top-sites-table', 'rows'),
    [dash.dependencies.Input('sites-date-range', 'start_date'),
     dash.dependencies.Input('sites-date-range', 'end_date')])
def update_comp_graph_slider(start_date, end_date):
    global results_df
    if(start_date is None or end_date is None):
        filtered_df = results_df
    else:
        filtered_df = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

    global global_filtered_top_sites_df
    global_filtered_top_sites_df = filtered_df

    sites_list = filtered_df['Sites'].apply(pd.Series).stack().reset_index(drop=True)
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


    return top_sites_df.to_dict('rows')

@app.callback(
    Output('other-view-selected', 'className'),
    [Input('other-sites-table', 'selected_row_indices')])
def disable_site_modal_button(selected_row_indices):
    if selected_row_indices:
        return 'view-selected-data'
    else:
        return 'view-selected-data-disabled'

@app.callback(
    Output('other-modal-site-table', 'rows'),
    [Input('other-view-selected', 'n_clicks')],
    [State('other-sites-table', 'rows'),
     State('other-sites-table', 'selected_row_indices')])
def update_site_modal_table(clicks, rows, selected_row_indices):
    print('here', clicks)

    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    print(sites)
    if(clicks):
        # ids = list(d['customdata'] for d in selectedData['points'])
        global global_filtered_other_sites_df

        dff = global_filtered_other_sites_df[global_filtered_other_sites_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global global_site_modal_ids
        global_site_modal_ids = list(dff['Response ID'])
        print(global_site_modal_ids)
        global global_other_selected_sites
        global_other_selected_sites = sites
        return dff.to_dict('rows')
    return []

@app.callback(Output('other-modal-site', 'style'),
              [Input('other-close-modal-site', 'n_clicks'),
               Input('other-view-selected', 'n_clicks')],
              [State('other-sites-table', 'selected_row_indices')])
def display_modal(closeClicks, openClicks, selected_row_indices):
    global otherSiteCloseCount
    if len(selected_row_indices) == 0:
        return {'display': 'none'}
    elif closeClicks > otherSiteCloseCount:
        otherSiteCloseCount = closeClicks
        return {'display': 'none'}
    elif openClicks:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('other_sites_slider_output', 'children'),
    [dash.dependencies.Input('other_sites_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)

@app.callback(
    dash.dependencies.Output('other-sites-table', 'rows'),
    [dash.dependencies.Input('sites-date-range', 'start_date'),
     dash.dependencies.Input('sites-date-range', 'end_date')])
def update_comp_graph_slider(start_date, end_date):
    global results_df
    if(start_date is None or end_date is None):
        filtered_df = results_df
    else:
        filtered_df = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

    global global_filtered_other_sites_df
    global_filtered_other_sites_df = filtered_df

    sites_list = filtered_df['Sites'].apply(pd.Series).stack().reset_index(drop=True)
    sites_list = ','.join(sites_list).split(',')
    sites_list = [x for x in sites_list if '.' in x]



    sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
    sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
    sites_df = sites_df.sort_values(by=['Count'])
    sites_df = sites_df[sites_df['Site'] != 'None Found']

    other_sites_list = [item for item in sites_list if item not in top_sites and item != 'None Found']
    other_sites_df = pd.DataFrame.from_dict(Counter(other_sites_list), orient='index').reset_index()
    other_sites_df = other_sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    other_sites_df = other_sites_df[other_sites_df['Count'] > 1]
    other_sites_df = other_sites_df.sort_values(by=['Count'], ascending=False)


    return other_sites_df.to_dict('rows')

# @app.callback(
#     Output('other-modal-site-table', 'rows'),
#     [Input('other-mentioned-site-graph', 'selectedData')])
# def update_site_modal_table(selectedData):
#     print('here', selectedData)
#     if(selectedData):
#         # ids = list(d['customdata'] for d in selectedData['points'])
#         sites = list(d['customdata'] for d in selectedData['points'])
#         dff = results_df[results_df['Sites'].isin(sites)]
#         cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         dff = dff[cnames]
#         dff.columns = cnamesnew
#         global global_site_modal_ids
#         global_site_modal_ids = list(dff['Response ID'])
#         print(global_site_modal_ids)
#         global global_selected_sites
#         global_selected_sites = sites
#         return dff.to_dict('rows')
#     return []


# @app.callback(Output('other-modal-site', 'style'),
#               [Input('other-close-modal-site', 'n_clicks'),
#                Input('other-mentioned-site-graph', 'selectedData')])
# def display_modal(closeClicks, selectedData):
#     global siteCloseCount
#     print('herehere', closeClicks, siteCloseCount)
#     if closeClicks > siteCloseCount:
#         siteCloseCount = closeClicks
#         return {'display': 'none'}
#     elif selectedData:
#         return {'display': 'block'}
#     else:
#         return {'display': 'none'}


@app.callback(
    Output('top-download-sites-link', 'href'),
    [Input('top-view-selected', 'n_clicks')],
    [State('top-sites-table', 'rows'),
     State('top-sites-table', 'selected_row_indices')])
def update_sites_download_link(clicks, rows, selected_row_indices):
    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    print(sites)
    if(clicks):
        # ids = list(d['customdata'] for d in selectedData['points'])
        global global_filtered_top_sites_df

        dff = global_filtered_top_sites_df[global_filtered_top_sites_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    return ''


@app.callback(
    Output('other-download-sites-link', 'href'),
    [Input('other-view-selected', 'n_clicks')],
    [State('other-sites-table', 'rows'),
     State('other-sites-table', 'selected_row_indices')])
def update_sites_download_link(clicks, rows, selected_row_indices):
    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    if(clicks):
        # ids = list(d['customdata'] for d in selectedData['points'])
        global global_filtered_other_sites_df

        dff = global_filtered_other_sites_df[global_filtered_other_sites_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    return ''


@app.callback(
    dash.dependencies.Output('comp_slider_output', 'children'),
    [dash.dependencies.Input('comp_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    dash.dependencies.Output('comp-graph', 'figure'),
    [dash.dependencies.Input('comp_time_slider', 'value')])
def update_comp_graph_slider(value):
    fig_component = updateGraph(component_df, 'Components Over Time', value)
    return fig_component


@app.callback(
    dash.dependencies.Output('list_comp_slider_output', 'children'),
    [dash.dependencies.Input('list_comp_time_slider', 'value')])
def update_list_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    dash.dependencies.Output('list-comp-graph', 'figure'),
    [dash.dependencies.Input('list_comp_time_slider', 'value')])
def update_list_comp_graph_slider(value):
    fig_component = updateGraph(list_component_df, 'Components Over Time', value)
    return fig_component


@app.callback(
    dash.dependencies.Output('issue_slider_output', 'children'),
    [dash.dependencies.Input('issue_time_slider', 'value')])
def update_issue_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    dash.dependencies.Output('issue-graph', 'figure'),
    [dash.dependencies.Input('issue_time_slider', 'value')])
def update_issue_graph_slider(value):
    fig_issue = updateGraph(issue_df, 'Issues Over Time', value)
    return fig_issue


@app.callback(
    dash.dependencies.Output('list_issue_slider_output', 'children'),
    [dash.dependencies.Input('list_issue_time_slider', 'value')])
def update_list_issue_output_slider(value):
    print('he')
    return 'Past {} days of data'.format(value)


@app.callback(
    dash.dependencies.Output('list-issue-graph', 'figure'),
    [dash.dependencies.Input('list_issue_time_slider', 'value')])
def update_list_issue_graph_slider(value):
    fig_issue = updateGraph(list_issue_df, 'Issues Over Time', value)
    return fig_issue


@app.callback(
    Output('modal-geo-table', 'rows'),
    [Input('country-graph', 'clickData')])
def update_site_modal_table(clickData):
    if(clickData):
        # ids = list(d['customdata'] for d in selectedData['points'])
        selected_geo = clickData['points'][0]['text']
        dff = results_df[results_df['Country'] == selected_geo]
        global global_geo_modal_ids
        global_geo_modal_ids = list(dff['Response ID'])

        cnames = ['Feedback', 'Date Submitted', 'Country',
                    'Components', 'Issues', 'Sites']
        cnamesnew = ['Feedback', 'Date Submitted', 'Country',
                   'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global global_selected_geo
        global_selected_geo = selected_geo
        return dff.to_dict('rows')
    return []

@app.callback(Output('modal-geo', 'style'),
              [Input('close-geo-modal', 'n_clicks'),
               Input('country-graph', 'clickData')])
def display_modal(closeClicks, openClick):
    global geoCloseCount
    if closeClicks > geoCloseCount:
        geoCloseCount = closeClicks
        return {'display': 'none'}
    elif openClick:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output('download-geo-link', 'href'),
    [Input('country-graph', 'clickData')])
def update_comp_download_link(clickData):
    if clickData:
        selected_geo = clickData['points'][0]['text']
        dff = search_df[search_df['Country'] == selected_geo]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ''

@app.callback(
    Output('geo_time_slider', 'disabled'),
    [Input('geoview-radio', 'value')])
def abling_slider(radio_value):
    if radio_value == 'week':
        return False
    else:
        return True

@app.callback(
    dash.dependencies.Output('geo_slider_output', 'children'),
    [dash.dependencies.Input('geo_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)

@app.callback(
    Output('country-graph', 'figure'),
    [Input('geoview-radio', 'value'),
     Input('geo_time_slider', 'value')])
def update_geoview_graph(radio_value, slider_value):
    fig_geo = updateGeoGraph(df_geo_sentiment, radio_value, slider_value)
    return fig_geo

# @app.callback(
#     dash.dependencies.Output('slider-container', 'children'),
#     [dash.dependencies.Input('sites-date-range', 'start_date'),
#      dash.dependencies.Input('sites-date-range', 'end_date')])
# def update_site_count(start_date, end_date):    #update graph with values that are in the time range
#     global results_df
#     if(start_date is None or end_date is None):
#         filtered_results = results_df
#     else:
#         filtered_results = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

#     sites_list = filtered_results['Sites'].apply(pd.Series).stack().reset_index(drop=True)
#     sites_list = ','.join(sites_list).split(',')
#     sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
#     sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
#     sites_df = sites_df[sites_df['Site'] != 'None Found']
#     count = len(sites_df.index)

#     return html.Div([
#         html.Div(dcc.Slider(
#             id='sites-slider',
#             min=0,
#             max=count,
#             value=count,
#             marks={i: '{} sites'.format(i) for i in range(count) if (i % 30 == 0) or (i == count)}),
#             style={'height': '50px', 'width': '100%', 'display': 'inline-block'}),
#         html.Div(id='sites-slider-output')
#     ])


@app.callback(
    dash.dependencies.Output('unique-site-count', 'children'),
    [dash.dependencies.Input('sites-date-range', 'start_date'),
     dash.dependencies.Input('sites-date-range', 'end_date')])
def update_site_count(start_date, end_date):    #update graph with values that are in the time range
    global results_df
    if(start_date is None or end_date is None):
        filtered_results = results_df
    else:
        filtered_results = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

    sites_list = filtered_results['Sites'].apply(pd.Series).stack().reset_index(drop=True)
    sites_list = ','.join(sites_list).split(',')
    sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
    sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    no_sites_df = sites_df[sites_df['Site'] == 'None Found']
    sites_df = sites_df[sites_df['Site'] != 'None Found']
    count = len(sites_df.index)

    return html.Div([
        html.P(['Sites were mentioned {} times in the raw feedback. There were {} unique sites mentioned.'.format(sites_df['Count'].sum(), count)]),
        html.P(['There were {} pieces of raw feedback that did not mention any sites.'.format(no_sites_df['Count'].sum())])
    ])

# @app.callback(
#     dash.dependencies.Output('sites-slider-output', 'children'),
#     [dash.dependencies.Input('sites-slider', 'value')])
# def update_output(value):
#     return 'Displaying {} most frequent sites.'.format(value)


# @app.callback(
#     dash.dependencies.Output('mentioned-site-graph', 'figure'),
#     [dash.dependencies.Input('sites-date-range', 'start_date'),
#      dash.dependencies.Input('sites-date-range', 'end_date'),
#      dash.dependencies.Input('sites-slider', 'value')])
# def update_site_graph(start_date, end_date, max):    #update graph with values that are in the time range
#     global results_df

#     if(start_date is None or end_date is None):
#         filtered_results = results_df
#     else:
#         filtered_results = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

#     sites_list = filtered_results['Sites'].apply(pd.Series).stack().reset_index(drop=True)
#     sites_list = ','.join(sites_list).split(',')
#     sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
#     sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
#     sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
#     sites_df = sites_df.sort_values(by=['Count'], ascending=False)
#     sites_df = sites_df[sites_df['Site'] != 'None Found']
#     sites_df = sites_df.head(max).sort_values(by='Count')

#     data = [go.Bar(
#         x=top_sites_df['Count'],
#         y=top_sites_df['Site'],
#         orientation='h',
#         customdata=top_sites_df['Site'],
#         marker=dict(
#             color='#E3D0FF'
#         ),
#     )]

#     layout = go.Layout(
#         title='Feedback by Mentioned Site(s)',
#         titlefont=dict(
#             family='Helvetica Neue, Helvetica, sans-serif',
#             color='#BCBCBC'
#         ),
#         xaxis=dict(
#             # showticklabels=False,
#             title='Number of Feedback'
#         ),
#         yaxis=dict(
#             title='Website'
#         ),
#         font=dict(
#             family='Helvetica Neue, Helvetica, sans-serif',
#             size=12,
#             color='#BCBCBC'
#         ),
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#     )

#     fig = dict(data=data, layout=layout)
#     return fig


@app.callback(
    Output('searchtable', 'rows'),
    [Input('searchrequest', 'n_submit')], # [Input('searchrequest', 'n_submit')],
    [State('searchrequest', 'value')])
def update_table(ns, request_value):
    df = search_df
    # cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
    #           'Feedback', 'Components', 'Issues', 'Sites']
    cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
              'Feedback', 'Components', 'Issues', 'Sites', 'Version']
    r_df = pd.DataFrame()
    # r_df = pd.DataFrame([cnames], columns=cnames)
    for index, row in df.iterrows():
        together = [str(row['Feedback']), str(row['Country']),
                    str(row['Components']), str(row['Issues']), str(row['Sites'])]
        fb = ''.join(together).lower()
        rv = str(request_value).lower()
        isit = rv in fb
        if isit:
            vers = re.search(r'Firefox/\s*([\d.]+)', str(row['User Agent']))
            # temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), int(row['compound']),
            #         str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])]
            # print(vers.group())
            table_vers=''
            if vers is None:
                table_vers=''
            else:
                table_vers = str(vers.group())
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), int(row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']),
                    str(row['Sites']), table_vers]
            temp_df = pd.DataFrame([temp], columns=cnames)
            r_df = r_df.append(temp_df, ignore_index=True)
    return r_df.to_dict('rows')


@app.callback(
    Output('search-download-link', 'href'),
    [Input('searchrequest', 'n_submit')], # [Input('searchrequest', 'n_submit')],
    [State('searchrequest', 'value')])
def update_table(ns, request_value):
    df = search_df
    # cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
    #           'Feedback', 'Components', 'Issues', 'Sites']
    cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
              'Feedback', 'Components', 'Issues', 'Sites', 'Version']
    r_df = pd.DataFrame()
    # r_df = pd.DataFrame([cnames], columns=cnames)
    for index, row in df.iterrows():
        together = [str(row['Feedback']), str(row['Country']),
                    str(row['Components']), str(row['Issues']), str(row['Sites'])]
        fb = ''.join(together).lower()
        rv = str(request_value).lower()
        isit = rv in fb
        if isit:
            # temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), int(row['compound']),
            #         str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])]
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), int(row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites']),
                    str(row['User Agent'])]
            temp_df = pd.DataFrame([temp], columns=cnames)
            r_df = r_df.append(temp_df, ignore_index=True)
    csv_string = r_df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


@app.callback(
    Output('search-count-reveal','children'),
    [Input('searchtable', 'rows')])
def set_search_count(dict_of_returned_df):
    if (len(dict_of_returned_df) > 0):
        if (len(dict_of_returned_df[0]) > 0):
            df_to_use = pd.DataFrame.from_dict(dict_of_returned_df)
            count = len(df_to_use.index)
            return u'Search returned {} results.'.format(count)
        else:
            return ''
    else:
        return u'Search returned no results.'

@app.callback(
    Output('search-loading', 'style'),
    [Input('search-table-container', 'style')],
    [State('searchrequest', 'value')])
def hide_loading(style, query):
    print(query)
    print(style['display'])
    if query and style['display'] == 'none':
        print('block')
        return {'display': 'block'}
    else:
        print('noneeeee')
        # return {'display': 'none'}
        return 'display: none'

@app.callback(
    Output('search-table-container','style'),
    [Input('search-count-reveal', 'children'),
     Input('searchtable', 'rows')])
def set_search_count(sentence, dict):
    if (len(dict[0]) > 0):
        return {'display': 'block',
                'animation': 'opac 1s'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('search-download-link','style'),
    [Input('search-count-reveal', 'children'),
     Input('searchtable', 'rows')])
def set_search_count(sentence, dict):
    if (len(dict[0]) > 0):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# @app.callback(
#     Output('download-search-link', 'href'),
#     [Input('searchtable', 'rows')])  #  https://github.com/plotly/dash-table/blob/master/dash_table/DataTable.py
# def update_search_download_link(rows):
#     dicttouse = dict()
#     print('we are doing this callback whoo')
#     print(type(rows))
#     print(len(rows))
#     print(rows)
#     for row in rows:
#         sampledict = ast.literal_eval(row)
#         dicttouse.update(sampledict)
#     print('we are doing this callback whoo')
#     print(dicttouse)
#     if rows:
#         sites = list(d['customdata'] for d in rows['points'])
#         dff = search_df[search_df['Sites'].isin(sites)]
#         cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
#                   'Feedback', 'Components', 'Issues', 'Sites']
#         cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
#                      'Feedback', 'Components', 'Issues', 'Sites']
#         dff = dff[cnames]
#         csv_string = dff.to_csv(index=False, encoding='utf-8')
#         csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
#         print(csv_string)
#         return csv_string
#     else:
#         return ''


if __name__ == '__main__':
    app.run_server(debug=False)

