import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table_experiments as dte
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
import ast
import json
from clustering import runDrilldown
from datetime import datetime as datetime
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE
from collections import Counter
import numpy as np
import urllib.parse

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_scripts = ['https://code.jquery.com/jquery-3.2.1.min.js']


# Reading in data:
results_df = pd.read_csv("./data/output_pipeline.csv", encoding ="ISO-8859-1")
results2_df = pd.read_csv("./data/output_pipeline.csv", encoding="ISO-8859-1")
sr_df = pd.read_csv("./data/output_spam_removed.csv", encoding="ISO-8859-1")
# print(results_df.shape) # SHOULD FILL NAN VALS AS WELL WHEN POSSIBLE
# search_df = results_df[["Response ID", "Date Submitted", "Country","City"\
#                         , "State/Region", "Binary Sentiment", "Positive Feedback"\
#                         , "Negative Feedback", "Relevant Site", "compound"\
#                         , "Sites", "Issues", "Components"]]
search_df = results_df
for index, row in search_df.iterrows():
    if pd.isnull(row['Sites']):
        search_df.at[index, 'Sites'] = 'None Found'
df = pd.read_csv('./data/output_countries.csv')
df1 = pd.read_csv('./data/Issues_Keywords_Clusters.csv', encoding='latin-1')
component_df = pd.read_csv('./data/component_graph_data.csv')
issue_df = pd.read_csv('./data/issue_graph_data.csv')
clusterDesc = pd.read_csv('./data/manual_cluster_descriptions.csv')
clusters_df = pd.read_csv('./data/output_clusters_defined.csv', usecols = ['Response ID', 'manual_clusters'])
global_site_modal_ids = []
global_selected_sites = []
siteCloseCount=0

# Getting components and issues in string:
WORDS_TO_COMPONENT = {k:(map(lambda word: word.lower(), v)) for k, v in WORDS_TO_COMPONENT.items()}
WORDS_TO_ISSUE = {k:(map(lambda word: word.lower(), v)) for k, v in WORDS_TO_ISSUE.items()}


# Setting data and layout for world map:
data = [ dict(
        type = 'choropleth',
        locations = df['CODE'],
        z = df['Sentiment'],
        text = df['COUNTRY'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
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
      )]


layout = dict(
    # title = 'This Week in Overall Global Sentiment of Mozilla Web Compat',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        ),
        bgcolor='rgba(0,0,0,0)',
    ),
    legend = dict (
        font = dict(
            family='Helvetica Neue, Helvetica, sans-serif',
            size=12,
            color='#D3D3D3'
        ),
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)


fig_geo = dict(data=data, layout=layout)


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
    'min': 1,
    'max': 14,
    'step': 1,
    'default': 7,
    'marks': {
        1: 1,
        7: 7,
        14: 14
    }
}
# GLOBALLY ADD DAY DIFFERENCE TO RESULTS DATAFRAME
reference = datetime(2016, 12, 30)
# reference = datetime.now()
results2_df['Day Difference'] = (reference - pd.to_datetime(results2_df['Date Submitted'], format='%Y-%m-%d %H:%M:%S')).dt.days + 1


def initCompDF(results2_df, num_days_range = 14):
    date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
    date_filtered_df['Components'] = date_filtered_df['Components'].apply(
        lambda x: ast.literal_eval(x))  # gives warning but works, fix later

    component_df = pd.Series([])
    comp_response_id_map = dict()

    for day in range(num_days_range):
        day_df = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]
        if(day_df.empty):
            continue
        # count docs with components
        new_comp_info = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
            lambda x: pd.Series(x).value_counts()).sum()
        # count docs with no assigned components
        if(0 in date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
                                       lambda x: len(x)).value_counts().index):
            new_comp_info = pd.concat([new_comp_info, 
                                        date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
                                        lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])
        
        component_df = pd.concat([component_df, new_comp_info.rename(str(day_df['Date Submitted'].values[0]).split(' ')[0])], axis=1)
        comp_response_id_map['Day ' + str(day + 1)] = dict()
        comps = new_comp_info.index.values
        for comp in comps:
            comp_response_id_map['Day ' + str(day + 1)][comp] = []

        for index, row in day_df.iterrows():
            for comp in row['Components']:
                comp_response_id_map['Day ' + str(day + 1)][comp].append(
                    row['Response ID'])  # TODO: can use map functions to make this faster
            if len(row['Components']) == 0 and 'No Label' in comps:
                comp_response_id_map['Day ' + str(day + 1)]['No Label'].append(row['Response ID'])

    component_df = component_df.fillna(0).astype(int).drop(0, 1).rename_axis('Components')
    return component_df, comp_response_id_map


def initIssueDF(results2_df, num_days_range = 14):
    date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
    date_filtered_df['Issues'] = date_filtered_df['Issues'].apply(lambda x: ast.literal_eval(x))

    issue_df = pd.Series([])
    issue_response_id_map = dict()

    for day in range(num_days_range):
        day_df = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]
        if(day_df.empty):
            continue
        new_issue_info = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
            lambda x: pd.Series(x).value_counts()).sum()
        # count docs with no assigned components
        if(0 in date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
                                       lambda x: len(x)).value_counts().index):
            new_issue_info = pd.concat([new_issue_info,
                                        date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
                                        lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])
        
        issue_df = pd.concat([issue_df, new_issue_info.rename(str(day_df['Date Submitted'].values[0]).split(' ')[0])], axis=1)

        issue_response_id_map['Day ' + str(day + 1)] = dict()
        issues = new_issue_info.index.values
        for issue in issues:
            issue_response_id_map['Day ' + str(day + 1)][issue] = [];

        for index, row in day_df.iterrows():
            for issue in row['Issues']:
                issue_response_id_map['Day ' + str(day + 1)][issue].append(row['Response ID'])
            if len(row['Issues']) == 0 and 'No Label' in issues:
                issue_response_id_map['Day ' + str(day + 1)]['No Label'].append(row['Response ID'])
    # Fill in component and issue df with 0 for Nan (?)
    issue_df = issue_df.fillna(0).astype(int).drop(0, 1).rename_axis('Issues')
    return issue_df, issue_response_id_map


def updateGraph(df, title, num_days_range = 7):
    filtered_df = df.iloc[:, 0:num_days_range]
    traces = []
    # Checking df for values:
    for index, row in filtered_df.iterrows():
        # print(list(row.keys()))
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


# CREATE FIRST TWO GRAPHS
day_range = min(results2_df['Day Difference'].max(), toggle_time_params['max'])
component_df, comp_response_id_map = initCompDF(results2_df, day_range)
list_component_df = component_df
issue_df, issue_response_id_map = initIssueDF(results2_df, day_range)
list_issue_df = issue_df
fig_component = updateGraph(component_df, 'Components Over Time', 7)
fig_issue = updateGraph(issue_df, 'Issues Over Time', 7)


def mergedGraph():
    # merge output_pipeline with output_clusters_defined
    merged = pd.merge(results_df, clusters_df, on='Response ID')
    merged = merged[merged['manual_clusters'].notna()]
    return merged


def updateCompMetricsGraph():
    # CATEGORIZATION VISUALIZATION
    merged = mergedGraph()
    compCountSeries = pd.Series([])
    # For components labelled:
    for component in WORDS_TO_COMPONENT.keys():
        compCounts = merged[merged['Components'].str.contains(component)]['manual_clusters'].value_counts()
        compCountSeries = pd.concat([compCountSeries, compCounts.rename(component)], axis=1)
    compCountSeries = pd.concat([compCountSeries, merged[merged['Components'].str.match("\[\]")][
        'manual_clusters'].value_counts().rename('No Label')], axis=1)
    compCountSeries = compCountSeries.drop(0, 1).fillna(0).astype(int)
    compCountSeries = compCountSeries.rename(index=categoryDict)
    traces_comp_metrics = []
    for index, row in compCountSeries.iterrows():
        # print(list(row.keys()))
        traces_comp_metrics.append(go.Bar(
            x=list(row.keys()),
            y=row.values,
            name=index,
            # hoverinfo='none',
            # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
            # customdata=docs.iloc[0].values
        ))
    def update_point(trace):
        # print(trace)
        return
    # Stacked Bar Graph figure - components labelled against manual labelling:
    layout_comp_metrics = go.Layout(
        barmode='stack',
        title='Components vs Manual Clusters',
        font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
        xaxis=dict(
            # showticklabels=False,
            title='Components'
        ),
        yaxis=dict(
            title='Count of Docs'
        )
    )
    fig_comp_metrics = dict(data=traces_comp_metrics, layout=layout_comp_metrics)
    return fig_comp_metrics


def updateIssuesMetricsGraph():
    # ISSUES VISUALIZATION
    merged = mergedGraph()
    # For issues labelled:
    issueCountSeries = pd.Series([])
    for issue in WORDS_TO_ISSUE.keys():
        issueCounts = merged[merged['Issues'].str.contains(issue)]['manual_clusters'].value_counts()
        issueCountSeries = pd.concat([issueCountSeries, issueCounts.rename(issue)], axis=1)
    issueCountSeries = pd.concat([issueCountSeries, merged[merged['Components'].str.match("\[\]")][
        'manual_clusters'].value_counts().rename('No Label')], axis=1)
    issueCountSeries = issueCountSeries.drop(0, 1).fillna(0).astype(int)
    issueCountSeries = issueCountSeries.rename(index=categoryDict)
    traces_issue_metrics = []
    for index, row in issueCountSeries.iterrows():
        # print(list(row.keys()))
        traces_issue_metrics.append(go.Bar(
            x=list(row.keys()),
            y=row.values,
            name=index,
            # hoverinfo='none',
            # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
            # customdata=docs.iloc[0].values
        ))
    # Stacked Bar Graph figure - issues labelled against manual labelling:
    layout_issue_metrics = go.Layout(
        barmode='stack',
        title='Issues vs Manual Clusters',
        font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
        xaxis=dict(
            # showticklabels=False,
            title='Issues'
        ),
        yaxis=dict(
            title='Count of Docs'
        )
    )
    fig_issue_metrics = dict(data=traces_issue_metrics, layout=layout_issue_metrics)
    return fig_issue_metrics


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
    words = list(df.loc['Words'].values)
    phrases = list(df.loc['Phrases'].values)
    traces = [go.Bar(
            x=words,
            y=count,
            text = phrases,
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
unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()


#compacted list of all sites mentioned in the comments
sites_list = results_df['Sites'].apply(pd.Series).stack().reset_index(drop=True)
sites_list = ','.join(sites_list).split(',')
sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
sites_df = sites_df.sort_values(by=['Count'], ascending=False)
sites_df = sites_df[sites_df['Site'] != 'None Found']


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
tabs_styles = {
    'height': '44px',
    'width': '600px',
    'display': 'inline-block'
}
tab_style = {
    # 'borderBottom': '1px solid #d6d6d6',
    'margin': '5px 0px 5px 0px',
    'padding': '11px',
    'backgroundColor': 'rgb(30,30,30)',
    'border': 'none',
}
sites_tab_style = {
    # 'borderBottom': '1px solid #d6d6d6',
    'margin': '5px 0px 5px 0px',
    'padding': '11px 14px 11px 14px',
    'backgroundColor': 'rgb(30,30,30)',
    'font-weight': 'bold',
    'border-style': 'solid',
    'border-width': '1px',
}
tab_selected_style = {
    'border': 'none',
    # 'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': 'rgb(30,30,30)',
    'color': 'white',
    'padding': '11px'
}
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.config.suppress_callback_exceptions = True
# app.layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='header',
#              children=[
#                  html.Div(id='left-header-container',
#                           children=[
#                               html.Img(id='logo', src='../assets/Mozilla-Firefox-icon.png'),
#                               html.H1(
#                                   children='Mozilla Customer Analytics',
#                                   id="title",
#                               ),
#                           ]),
#                  dcc.Tabs(id="tabs-styled-with-inline", value='/sites', children=[
#                      dcc.Tab(label='Sentiment', value='/sentiment', style=tab_style, selected_style=tab_selected_style),
#                      dcc.Tab(label='Geo-View', value='/geoview', style=tab_style, selected_style=tab_selected_style),
#                      dcc.Tab(label='Components', value='/components', style=tab_style,
#                              selected_style=tab_selected_style),
#                      dcc.Tab(label='Issues', value='/issues', style=tab_style, selected_style=tab_selected_style),
#                      dcc.Tab(label='SITES', value='/sites', style=sites_tab_style, selected_style=tab_selected_style),
#                      dcc.Tab(label='Search', value='/search', style=tab_style, selected_style=tab_selected_style),
#                  ], style=tabs_styles),
#              ]),
#     html.H3('   '),  # Need vertical space for the tabs to not be overlapped by the page content
#     html.Div(id='test-div'),
#     html.Div(id='page-content')
# ])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

main_layout = html.Div(children=[
    html.Div(id="header",
             children=[
        html.H1(
            children='Mozilla Web Compat Analytics',
            id="title",
        ),
        dcc.Tabs(id="tabs-styled-with-inline", value='sites', children=[
            dcc.Tab(label='Sentiment', value='sentiment', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Geo-View', value='geoview', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Components', value='components', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Issues', value='issues', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='SITES', value='sites', style=tab_style, selected_style=tab_selected_style),
            dcc.Tab(label='Search', value='search', style=tab_style, selected_style=tab_selected_style),
        ], style=tabs_styles),
    ]),
    html.Div(id='tabs-content-inline'),
])


#prep data for displaying in stacked binary sentiment graph over time
#Grab unique dates from results_df
results_df["Date Submitted"] = pd.to_datetime(results_df["Date Submitted"])
unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()
common_df = test2 = results_df.groupby('Sites')['Sites'].agg(['count']).reset_index()
common_df = common_df.sort_values(by=['count'], ascending=False)



list_page_children = []


# sites_layout = html.Div([
#     html.H2('Sites'),
#     html.Div([
#         html.Label('Choose Date Range:'),
#         dcc.DatePickerRange(
#             id='sites-date-range',
#             min_date_allowed=results_df['Date Submitted'].min(),
#             max_date_allowed=results_df['Date Submitted'].max(),
#             start_date=results_df['Date Submitted'].min(),
#             end_date=results_df['Date Submitted'].max()
#         )
#     ]),
#
#     dcc.Graph(
#         id='mentioned-site-graph',
#         figure={
#             'data': [{
#                 'x': common_df[common_df.columns[0]],
#                 'y': common_df[common_df.columns[1]],
#                 'customdata': results_df['Sites'].unique()[1:],
#                 'type': 'bar'
#             }],
#             'layout': {
#                 'title': "Feedback by Mentioned Site(s)",
#                 'xaxis': {
#                     'title': 'Mentioned Site(s)'
#                 },
#                 'yaxis': {
#                     'title': 'Number of Feedback'
#                 }
#             }
#         }
#     ),
#     dt.DataTable(
#         id='common-site-table',
#         columns=[{"name": i, "id": i} for i in search_df.columns],
#         pagination_settings={
#             'current_page': 0,
#             'page_size': PAGE_SIZE
#         },
#         pagination_mode='be',
#         sorting='be',
#         sorting_type='single',
#         sorting_settings=[],
#         n_fixed_rows=1,
#         style_table={
#             'overflowX': 'scroll',
#             'maxHeight': '800',
#             'overflowY': 'scroll'
#         },
#         style_cell={
#             'minWidth': '50'
#                         'px', 'maxWidth': '200px',
#             'whiteSpace': 'no-wrap',
#             'overflow': 'hidden',
#             'textOverflow': 'ellipsis',
#         },
#         style_cell_conditional=[
#             {
#                 'if': {'column_id': 'Feedback'},
#                 'textAlign': 'left'
#             }
#         ],
#         css=[{
#             'selector': '.dash-cell div.dash-cell-value',
#             'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',
#         }],
#     )
# ])
sites_layout = html.Div(className='sites-layout', children=[
    html.H3('Sites', className='page-title'),
    html.Div([
        html.Label('Choose Date Range:'),
        dcc.DatePickerRange(
            id='sites-date-range',
            min_date_allowed=results_df['Date Submitted'].min(),
            max_date_allowed=results_df['Date Submitted'].max(),
            start_date=results_df['Date Submitted'].min(),
            end_date=results_df['Date Submitted'].max()
        ),
        html.Div(id='unique-site-count')
    ]),
    html.Div(children=
        dcc.Graph(
            id='mentioned-site-graph',
            figure={
                'data': [{
                    'x': sites_df['Formatted'],
                    'y': sites_df['Count'],
                    'customdata': sites_df['Site'],
                    'type': 'bar',
                    'marker': {
                        'color': '#E3D0FF'
                    },
                }],
                'layout': {
                    'title': "Feedback by Mentioned Site(s)",
                    'titlefont': {
                        'family': 'Helvetica Neue, Helvetica, sans-serif',
                        'color': '#BCBCBC',
                    },
                    'xaxis': {
                        'title': ''
                    },
                    'yaxis': {
                        'title': 'Number of Feedback'
                    },
                    'font': {
                        'family': 'Helvetica Neue, Helvetica, sans-serif',
                        'size': 12,
                        'color': '#BCBCBC',
                    },
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                },
            },
        ),
    ),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Div(className="close-button-container", children=[
                html.Button("Close", id="close-modal-site", className="close", n_clicks=0),
            ]),
            html.P(""),
            html.H2("Selected Feedback Data Points", className='modal-title'),  # Header
            html.Div(className='drill-down-container', children=[
                html.A("Drill-down", className='drill-down-link', href='/list', target="_blank"), # close button
                html.A("Download CSV", id='download-link', className='download-link', href='', target="_blank"),
            ]),
            html.Div(className='modal-table-container', children=[
                dte.DataTable(  # Add fixed header row
                    id='modal-site-table',
                    rows=[{}],
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                ),
            ]),
        ], id='modal-content-site', className='modal-content')
    ], id='modal-site', className='modal'),
])


sentiment_layout = html.Div([
    html.H2('Sentiment'),
    # dcc.RadioItems(
    #     id='bin',
    #     options=[{'label': i, 'value': i} for i in [
    #         'Yearly', 'Monthly', 'Weekly', 'Daily'
    #     ]],
    #     value='Daily',
    #     labelStyle={'display': 'inline'}
    # ),
    html.Div([
        html.Div(
            className='six columns',
            children=dcc.Graph(
               id='binary-sentiment-ts',
               figure={
                        'data': [
                            {
                                'x': unique_dates,
                                'y': results_df[results_df["Binary Sentiment"] == "Sad"].groupby(
                                    [results_df['Date Submitted'].dt.date])['Binary Sentiment'].count().values,
                                'type': 'bar',
                                'name': "Sad"
                            },
                            {
                            'x': unique_dates,
                            'y': results_df[results_df["Binary Sentiment"] == "Happy"].groupby([results_df['Date Submitted'].dt.date])['Binary Sentiment'].count().values,
                            'type': 'bar',
                            'name': "Happy"
                            }
                        ],
                        'layout': {
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'barmode': 'stack',
                            'font': {
                                'color': colors['text']
                            }
                        }
                    }
                )
            ),
        html.Div(
            className='six columns',
            children=dcc.Graph(
                    id='trends-scatterplot',
                    figure={
                        'data': [{
                            'x': results_df['Date Submitted'],
                            'y': results_df['compound'],
                            'customdata': results_df['Response ID'],
                            'type': 'line',
                            'name': "Sentiment score",
                            'mode': 'markers',
                            'marker': {'size': 12}
                        }],
                        'layout': {
                             'title': "Compound Sentiment Score Over Time"
                        }
                    }
            )
        )
    ]),
    html.Div([
        html.Div(
            className='six columns',
            children=[
                # dcc.Graph(id='trend-data-histogram'),
                html.Button('Display Selected Data', id='display_data', n_clicks_timestamp=0)
            ]
        ),
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
                                 style_cell_conditional=[
                                     {
                                         'if': {'column_id': 'Feedback'},
                                         'textAlign': 'left'
                                     }
                                 ],
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
    # html.Div(id="bitch-div"),
    # html.Div(id="bitch-div2")
])


geoview_layout = html.Div([
    html.H2('Geographical View'),
    dcc.Graph(id='country-graph', figure=fig_geo),
    # html.Div(id="bitch-div"),
    # html.Div(id="bitch-div2")
])


components_layout = html.Div([
    # html.H2('Components'),
    # html.Div(id = 'comp_slider_output'),
    # dcc.Slider(id='comp_time_slider',
    #            min=toggle_time_params['min'], max=toggle_time_params['max'],
    #            step=toggle_time_params['step'], value=toggle_time_params['default'],
    #            marks=toggle_time_params['marks']),
    # dcc.Graph(id='graph2', figure=fig_component),
    html.Div(id='comp_container',
             children=[
                 html.Div(id='comp_slider_output'),
                 dcc.Slider(id='comp_time_slider',
                            min=toggle_time_params['min'], max=toggle_time_params['max'],
                            step=toggle_time_params['step'], value=toggle_time_params['default'],
                            marks=toggle_time_params['marks']),
                 dcc.Graph(id='comp-graph', figure=fig_component),
             ]
    ),
    html.Div(className='row', children=[
        html.Div([
            html.Div(id='click-data'),  # Doesn't do anything right now
        ]),
    ]),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Button("Close", id="close-modal-comp-issue", className="close", n_clicks_timestamp=0),
            html.A("Link to external site", href='/list', target="_blank"),  # close button
            html.H2("Selected Feedback Data Points"),  # Header
            dcc.Graph(id='modal-cluster-graph'),  # Clustering Bar Graph
            dte.DataTable(  # Add fixed header row
                id='modal-comp-issue-table',
                rows=[{}],
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
            ),
        ], id='modal-content-comp-issue', className='modal-content')
    ], id='modal-comp-issue', className='modal'),
])


issues_layout = html.Div([
    # html.H2('Issues'),
    # html.Div(id='issue_slider_output'),
    # dcc.Slider(id='issue_time_slider',
    #            min=toggle_time_params['min'], max=toggle_time_params['max'],
    #            step=toggle_time_params['step'], value=toggle_time_params['default'],
    #            marks=toggle_time_params['marks']),
    # dcc.Graph(id='graph3', figure=fig_issue),
    html.Div(id='issue_container',
             children=[
                 html.Div(id='issue_slider_output'),
                 dcc.Slider(id='issue_time_slider',
                            min=toggle_time_params['min'], max=toggle_time_params['max'],
                            step=toggle_time_params['step'], value=toggle_time_params['default'],
                            marks=toggle_time_params['marks']),
                 dcc.Graph(id='issue-graph', figure=fig_issue),
    ]),
    # dcc.Graph(id='graph4', figure=fig_comp_metrics),
    # dcc.Graph(id='graph5', figure=fig_issue_metrics),
    html.Div(className='row', children=[
        html.Div([
            html.Div(id='click-data'),  # Doesn't do anything right now
        ]),
    ]),
    html.Div([  # entire modal
        # modal content
        html.Div([
            html.Button("Close", id="close-modal-comp-issue", className="close", n_clicks_timestamp=0),
            html.A("Link to external site", href='/list', target="_blank"),  # close button
            html.H2("Selected Feedback Data Points"),  # Header
            dcc.Graph(id='modal-cluster-graph'),  # Clustering Bar Graph
            dte.DataTable(  # Add fixed header row
                id='modal-comp-issue-table',
                rows=[{}],
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
            ),
        ], id='modal-content-comp-issue', className='modal-content')
    ], id='modal-comp-issue', className='modal'),
])


search_layout = html.Div([
    html.H3('Search Feedback'),
    # html.Label('Enter Search Request:'),
    dcc.Input(id='searchrequest', type='text', placeholder='Search'),
    html.Div(id='search-table-container',
             children=[
                 dte.DataTable(  # Add fixed header row
                     id='searchtable',
                     rows=[{}],
                     row_selectable=True,
                     filterable=True,
                     sortable=True,
                     selected_row_indices=[],
                 ),
             ],
             style={'display': 'none'}),
    html.Div(id='search-count-reveal')
])


# @app.callback(
#     Output('bitch-div2', 'children'),
#     [Input('issue-graph', 'clickData')])
# def display_click_data(clickData):
#     if (len(clickData['points']) == 1):
#         day = clickData['points'][0]['x']
#         issue = clickData['points'][0]['customdata']
#         ids = issue_response_id_map[day][issue]
#         df = results_df[results_df['Response ID'].isin(ids)]
#         return df
#     else:
#         return ''



# @app.callback(dash.dependencies.Output('page-content', 'children'),
#               [dash.dependencies.Input('url', 'pathname')])
# def display_page(pathname):
#     print('current path', pathname)
#     # this is where you put the stuff
#     # set the data you need into a variable inside the click callback on the graph (the same place where you open the modal)
#     # reference that variable here to build the page contents
#     if pathname == '/list':
#         # global_sites_list_df is the dataframe that contains the data that appears in the modal
#         # ideally this should be fed through the same functions as results2_df to create the figures to display on the new page
#         global global_sites_list_df
#         global_sites_list_df['Day Difference'] = (reference - pd.to_datetime(global_sites_list_df['Date Submitted'],
#                                                                              format='%Y-%m-%d %H:%M:%S')).dt.days + 1
#         day_range_site_list = min(global_sites_list_df['Day Difference'].max(), toggle_time_params['max'])

#         # component_df_list, comp_response_id_map_list = initCompDF(global_sites_list_df, day_range_site_list)
#         # issue_df_list, issue_response_id_map_list = initIssueDF(global_sites_list_df, day_range_site_list)
#         fig_component_list = updateGraph(global_sites_list_df, 'Components Over Time', 7)
#         fig_issue_list = updateGraph(global_sites_list_df, 'Issues Over Time', 7)
#         return html.Div([
#             html.Div(id='comp_container',
#                      className='one-half column',
#                      children=[
#                          html.Div(id='comp_slider_output'),
#                          dcc.Slider(id='comp_time_slider',
#                                     min=toggle_time_params['min'], max=toggle_time_params['max'],
#                                     step=toggle_time_params['step'], value=toggle_time_params['default'],
#                                     marks=toggle_time_params['marks']),
#                          dcc.Graph(id='comp-graph', figure=fig_component_list),
#                      ]
#                      ),
#             html.Div(id='issue_container',
#                      className='one-half column',
#                      children=[
#                          html.Div(id='issue_slider_output'),
#                          dcc.Slider(id='issue_time_slider',
#                                     min=toggle_time_params['min'], max=toggle_time_params['max'],
#                                     step=toggle_time_params['step'], value=toggle_time_params['default'],
#                                     marks=toggle_time_params['marks']),
#                          dcc.Graph(id='issue-graph', figure=fig_issue_list),
#                      ]),
#         ])
#     elif pathname == '/sites':
#         return sites_layout
#     elif pathname == '/sentiment':
#         return sentiment_layout
#     elif pathname == '/geoview':
#         return geoview_layout
#     elif pathname == '/components':
#         return components_layout
#     elif pathname == '/issues':
#         return issues_layout
#     elif pathname == '/search':
#         return search_layout
#     else:
#         return sites_layout


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    # this is where you put the stuff
    # set the data you need into a variable inside the click callback on the graph (the same place where you open the modal)
    # reference that variable here to build the page contents
    if pathname == '/list':
        # global_sites_list_df is the dataframe that contains the data that appears in the modal
        # ideally this should be fed through the same functions as results2_df to create the figures to display on the new page
        global global_site_modal_ids
        global list_component_df
        global list_issue_df
        global global_selected_sites
        results_modal_df = results2_df[results2_df['Response ID'].isin(global_site_modal_ids)]
        day_range_site_list = min(results_modal_df['Day Difference'].max(), toggle_time_params['max'])

        component_df_list, comp_response_id_map_list = initCompDF(results_modal_df, day_range_site_list)
        list_component_df = component_df_list
        issue_df_list, issue_response_id_map_list = initIssueDF(results_modal_df, day_range_site_list)
        list_issue_df = issue_df_list
        fig_component_list = updateGraph(component_df_list, 'Components Over Time', 7)
        fig_issue_list = updateGraph(issue_df_list, 'Issues Over Time', 7)
        return html.Div([
            html.Div([
                html.H1(
                    children='Mozilla Web Compat Analytics',
                    id="title",
                ),
            ]),
            html.Div([
                html.Div(id='list_comp_container',
                         className='one-half column',
                         children=[
                             html.Div(id='list_comp_slider_output'),
                             dcc.Slider(id='list_comp_time_slider',
                                        min=toggle_time_params['min'], max=toggle_time_params['max'],
                                        step=toggle_time_params['step'], value=toggle_time_params['default'],
                                        marks=toggle_time_params['marks']),
                             dcc.Graph(id='list-comp-graph', figure=fig_component_list),
                         ]
                 ),
                html.Div(id='list_issue_container',
                         className='one-half column',
                         children=[
                             html.Div(id='list_issue_slider_output'),
                             dcc.Slider(id='list_issue_time_slider',
                                        min=toggle_time_params['min'], max=toggle_time_params['max'],
                                        step=toggle_time_params['step'], value=toggle_time_params['default'],
                                        marks=toggle_time_params['marks']),
                             dcc.Graph(id='list-issue-graph', figure=fig_issue_list),
                ]),
                ]),
            html.Ul([html.Li(x) for x in global_selected_sites])
        ])
    else:
        return main_layout


# @app.callback(Output('url', 'pathname'),
#               [Input('tabs-styled-with-inline', 'value')])
# def update_url(tab):  # bit of a hacky way of updating URL for now.
#     print("clicked tab", tab)
#     return tab

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


# # Show Comp/Issue Modal on click
@app.callback(Output('modal-comp-issue', 'style'),
              [Input('comp-graph', 'clickData'),
               Input('issue-graph', 'clickData')])
def display_modal(compClickData, issueClickData):
    if compClickData or issueClickData:
        print('block')
        return {'display': 'block'}
    else:
        return {'display': 'none'}


# Drilldown Clustering Bar Graph
@app.callback(Output('modal-cluster-graph', 'figure'),
              [Input('comp-graph', 'clickData'),
               Input('issue-graph', 'clickData')])
def display_modal(compClickData, issueClickData):
    if compClickData:
        clickData = compClickData

        if (len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            component = clickData['points'][0]['customdata']
            ids = comp_response_id_map[day][component]
            dff = sr_df[sr_df['Response ID'].isin(ids)]
        else:
            return
    elif issueClickData:
        clickData = issueClickData

        if (len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            issue = clickData['points'][0]['customdata']
            ids = issue_response_id_map[day][issue]
            dff = sr_df[sr_df['Response ID'].isin(ids)]
        else:
            return
    # fig = drilldownClustering(dff)
    return {}
# @app.callback(Output('modal-cluster-graph', 'figure'),
#               [Input('comp-graph', 'clickData'),
#                Input('issue-graph', 'clickData')])
# def display_modal(compClickData, issueClickData):
#     if compClickData:
#         clickData = compClickData
#
#         if (len(clickData['points']) == 1):
#             day = clickData['points'][0]['x']
#             component = clickData['points'][0]['customdata']
#             ids = comp_response_id_map[day][component]
#             dff = sr_df[sr_df['Response ID'].isin(ids)]
#         else:
#             return
#     elif issueClickData:
#         clickData = issueClickData
#
#         if (len(clickData['points']) == 1):
#             day = clickData['points'][0]['x']
#             issue = clickData['points'][0]['customdata']
#             ids = issue_response_id_map[day][issue]
#             dff = sr_df[sr_df['Response ID'].isin(ids)]
#         else:
#             return
#
#     fig = drilldownClustering(dff)
#
#     return fig

# Component Drilldown Data Table
@app.callback(
    Output('modal-table-comp-issue', 'data'),
    [Input('comp-graph', 'clickData'),
     Input('issue-graph', 'clickData'),
     Input('modal-table-comp-issue', "pagination_settings")])
def display_click_data(compClickData, issueClickData, pagination_settings):
    #Set click data to whichever was clicked
    if (compClickData):
        clickData = compClickData

        if (len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            component = clickData['points'][0]['customdata']
            ids = comp_response_id_map[day][component]
            dff = results_df[results_df['Response ID'].isin(ids)]

    elif(issueClickData):
        clickData = issueClickData

        if (len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            issue = clickData['points'][0]['customdata']
            ids = issue_response_id_map[day][issue]
            dff = results_df[results_df['Response ID'].isin(ids)]
    else:
        return ''

    return dff.iloc[
           pagination_settings['current_page'] * pagination_settings['page_size']:
           (pagination_settings['current_page'] + 1) * pagination_settings['page_size']
           ].to_dict('rows')



# Scatterplot Drilldown click
@app.callback(Output('modal', 'style'), [Input('display_data','n_clicks_timestamp'),
                                         Input('close-modal', 'n_clicks_timestamp')])
def display_modal(openm, closem):
    if closem > openm:
        return {'display': 'none'}
    elif openm > closem:
        return {'display': 'block'}
    else: 
        return {}


@app.callback(
    Output('modal-table', "data"),
    [Input('modal-table', "pagination_settings"),
     Input('modal-table', "sorting_settings"),
     Input('display_data','n_clicks_timestamp'),
     Input('close-modal', 'n_clicks_timestamp'),
     Input('trends-scatterplot', 'selectedData')])
def update_modal_table(pagination_settings, sorting_settings, openm, closem, selectedData):
    if openm > closem: # only update the table if the modal is open
        ids = list(d['customdata'] for d in selectedData['points'])
        dff = search_df[search_df['Response ID'].isin(ids)]

        if len(sorting_settings):
            dff = dff.sort_values(
                [col['column_id'] for col in sorting_settings],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sorting_settings
                ],
                inplace=False
            )

        return dff.iloc[
               pagination_settings['current_page'] * pagination_settings['page_size']:
               (pagination_settings['current_page'] + 1) * pagination_settings['page_size']
               ].to_dict('rows')
    else:
        return {}


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


@app.callback(
    Output('trend-data-histogram', 'figure'),
    [Input('trends-scatterplot', 'selectedData')])
def display_selected_trend_data(selectedData):
    # return table matching the current selection
    ids = list(d['customdata'] for d in selectedData['points'])
    df = search_df[search_df['Response ID'].isin(ids)]
    # print(ids)
    return {
        'data': [
            {
                'x': df['compound'],
                'name': 'Compound Sentiment',
                'type': 'histogram',
                'autobinx': True
            }
        ],
        'layout': {
            'margin': {'l': 40, 'r': 20, 't': 0, 'b': 30}
        }
    }


@app.callback(
    Output('modal-site-table', 'rows'),
    [Input('mentioned-site-graph', 'selectedData')])
def update_site_modal_table(selectedData):
    print('here', selectedData)
    if(selectedData):
        # ids = list(d['customdata'] for d in selectedData['points'])
        sites = list(d['customdata'] for d in selectedData['points'])
        dff = search_df[search_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global global_site_modal_ids
        global_site_modal_ids = list(dff['Response ID'])
        print(global_site_modal_ids)
        global global_selected_sites 
        global_selected_sites = sites
        return dff.to_dict('rows')
    return []


@app.callback(Output('modal-site', 'style'),
              [Input('close-modal-site', 'n_clicks'),
               Input('mentioned-site-graph', 'selectedData')])
def display_modal(closeClicks, selectedData):
    global siteCloseCount
    print('herehere', closeClicks, siteCloseCount)
    if closeClicks > siteCloseCount:
        siteCloseCount = closeClicks
        return {'display': 'none'}
    elif selectedData:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('download-link', 'href'),
    [Input('mentioned-site-graph', 'selectedData')])
def update_download_link(selectedData):
    if selectedData:
        sites = list(d['customdata'] for d in selectedData['points'])
        dff = search_df[search_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        print(csv_string)
        return csv_string
    else: 
        return ''


# Component DF Slider Callback
@app.callback(
    dash.dependencies.Output('comp_slider_output', 'children'),
    [dash.dependencies.Input('comp_time_slider', 'value')])
def update_output(value):
    return 'Past {} days of data'.format(value)


# Component DF Time Toggle Callback
@app.callback(
    dash.dependencies.Output('comp-graph', 'figure'),
    [dash.dependencies.Input('comp_time_slider', 'value')])
def update_output(value):
    fig_component = updateGraph(component_df, 'Components Over Time', value)
    return fig_component

@app.callback(
    dash.dependencies.Output('list_comp_slider_output', 'children'),
    [dash.dependencies.Input('list_comp_time_slider', 'value')])
def update_output(value):
    return 'Past {} days of data'.format(value)


# Component DF Time Toggle Callback
@app.callback(
    dash.dependencies.Output('list-comp-graph', 'figure'),
    [dash.dependencies.Input('list_comp_time_slider', 'value')])
def update_output(value):
    print('hehehe', value)
    fig_component = updateGraph(list_component_df, 'Components Over Time', value)
    return fig_component


divs = []


# Component DF Slider Callback
# @app.callback(
#     Output('click-data', 'children'),
#     [Input('graph2', 'clickData')])
# def display_click_data(clickData):
#     if (clickData):
#         htmlArr = []
#         data = clickData['points'][0]['customdata']
#         docData = json.loads(json.dumps(ast.literal_eval(data)))
#         for key, value in docData.items():
#             docArray = []
#             for doc in value:
#                 docArray.append(html.Div(doc, style={'outline': '1px dotted green'}))
#             htmlArr.append(
#                 html.Div([
#                     html.H4(key),
#                     html.Div(children=docArray)
#                 ])
#             )
#         return htmlArr
#     return ''


@app.callback(
    dash.dependencies.Output('issue_slider_output', 'children'),
    [dash.dependencies.Input('issue_time_slider', 'value')])
def update_output(value):
    return 'Past {} days of data'.format(value)


# Component DF Time Toggle Callback
@app.callback(
    dash.dependencies.Output('issue-graph', 'figure'),
    [dash.dependencies.Input('issue_time_slider', 'value')])
def update_output(value):
    fig_issue = updateGraph(issue_df, 'Issues Over Time', value)
    return fig_issue

@app.callback(
    dash.dependencies.Output('list_issue_slider_output', 'children'),
    [dash.dependencies.Input('list_issue_time_slider', 'value')])
def update_output(value):
    return 'Past {} days of data'.format(value)


# Component DF Time Toggle Callback
@app.callback(
    dash.dependencies.Output('list-issue-graph', 'figure'),
    [dash.dependencies.Input('list_issue_time_slider', 'value')])
def update_output(value):
    fig_issue = updateGraph(list_issue_df, 'Issues Over Time', value)
    return fig_issue


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
    sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
    sites_df = sites_df.sort_values(by=['Count'], ascending=False)
    no_sites_df = sites_df[sites_df['Site'] == 'None Found']
    sites_df = sites_df[sites_df['Site'] != 'None Found']
    count = len(sites_df.index)

    print('here in updateSiteCount', no_sites_df)

    return html.Div([
        html.P(['Sites were mentioned {} times in the raw feeback.'.format(sites_df['Count'].sum())]),
        html.P(['There were {} unique sites mentioned.'.format(count)]),
        html.P(['There were {} raw feedback with no mentions of sites.'.format(no_sites_df['Count'].sum())]),
    ])


@app.callback(
    dash.dependencies.Output('mentioned-site-graph', 'figure'),
    [dash.dependencies.Input('sites-date-range', 'start_date'),
     dash.dependencies.Input('sites-date-range', 'end_date')])
def update_site_graph(start_date, end_date):    #update graph with values that are in the time range
    print('here', start_date, end_date)

    global results_df

    if(start_date is None or end_date is None):
        print('herherherherheh')
        filtered_results = results_df
    else:
        filtered_results = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]


    print(filtered_results)

    sites_list = filtered_results['Sites'].apply(pd.Series).stack().reset_index(drop=True)
    sites_list = ','.join(sites_list).split(',')
    sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
    sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
    sites_df = sites_df.sort_values(by=['Count'], ascending=False)
    sites_df = sites_df[sites_df['Site'] != 'None Found']

    data = [go.Bar(
        x=sites_df['Formatted'],
        y=sites_df['Count'],
        customdata=sites_df['Site'],
        marker=dict(
            color='#E3D0FF'
        ),
    )]

    layout = go.Layout(
        title='Feedback by Mentioned Site(s)',
        titlefont=dict(
            family='Helvetica Neue, Helvetica, sans-serif',
            color='#BCBCBC'
        ),
        xaxis=dict(
            # showticklabels=False,
            title=''
        ),
        yaxis=dict(
            title='Number of Feedback'
        ),
        font=dict(
            family='Helvetica Neue, Helvetica, sans-serif',
            size=12,
            color='#BCBCBC'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
    )

    fig = dict(data=data, layout=layout)
    return fig



# @app.callback(
#     dash.dependencies.Output('mentioned-site-graph', 'figure'),
#     [dash.dependencies.Input('sites-date-range', 'start_date'),
#      dash.dependencies.Input('sites-date-range', 'end_date')])
# def update_graph_data(start_date, end_date):    #update graph with values that are in the time range
#     print("swag")
    # start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    # end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    # df = results_df[(results_df['Date Submitted'] >= start_date) & (results_df['Date Submitted'] <= end_date)].groupby('Sites')['Sites'].agg(['count']).reset_index()
    # data = {
    #     'x': df.columns[0],
    #     'y': df.columns[1],
    #   # 'customdata': results_df['Sites'].unique()[1:],
    #     'type': 'bar'
    # }

    # return data


# @app.callback(
#     Output('test-div', "children"),
#     [Input('mentioned-site-graph', "clickData")])
# def update_common_table(clickData):
#     # print(clickData)
#     print('CLICKED DATA', clickData)

#     # dff = search_df[search_df['Sites'] == clickData['points'][0]['customdata']]
#     # if len(sorting_settings):
#     #     dff = dff.sort_values(
#     #         [col['column_id'] for col in sorting_settings],
#     #         ascending=[
#     #             col['direction'] == 'asc'
#     #             for col in sorting_settings
#     #         ],
#     #         inplace=False
#     #     )

#     # return dff.iloc[
#     #        pagination_settings['current_page'] * pagination_settings['page_size']:
#     #        (pagination_settings['current_page'] + 1) * pagination_settings['page_size']
#     #        ].to_dict('rows')


@app.callback(
    Output('searchtable', 'rows'),
    [Input('searchrequest', 'n_submit'), Input('searchrequest', 'n_blur'),],
    [State('searchrequest', 'value')])
def update_table(ns, nb, request_value):
    df = search_df
    cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
              'Feedback', 'Components', 'Issues', 'Sites']
    r_df = pd.DataFrame()
    # r_df = pd.DataFrame([cnames], columns=cnames)
    for index, row in df.iterrows():
        together = [str(row['Feedback']), str(row['Country']),
                    str(row['Components']), str(row['Issues']), str(row['Sites'])]
        fb = ''.join(together).lower()
        rv = str(request_value).lower()
        isit = rv in fb
        if isit:
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), str(row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])]
            temp_df = pd.DataFrame([temp], columns=cnames)
            r_df = r_df.append(temp_df, ignore_index=True)
    return r_df.to_dict('rows')


# NEED TO FIX THIS
@app.callback(
    Output('search-count-reveal','children'),
    [Input('searchtable', 'rows')])
def set_search_count(dict_of_returned_df):
    df_to_use = pd.DataFrame.from_dict(dict_of_returned_df)
    count = len(df_to_use.index)
    return u'Search returned {} results.'.format(count)



if __name__ == '__main__':
    app.run_server(debug=True)

