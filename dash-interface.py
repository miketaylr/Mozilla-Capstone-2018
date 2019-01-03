import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table_experiments as dte
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
import clustering as clustering
import ast
import json
from datetime import datetime
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Reading in data:
results_df = pd.read_csv("./data/output_pipeline.csv", encoding ="ISO-8859-1")
results2_df = pd.read_csv("./data/output_pipeline.csv", encoding="ISO-8859-1")
# print(results_df.shape) # SHOULD FILL NAN VALS AS WELL WHEN POSSIBLE
# search_df = results_df[["Response ID", "Date Submitted", "Country","City"\
#                         , "State/Region", "Binary Sentiment", "Positive Feedback"\
#                         , "Negative Feedback", "Relevant Site", "compound"\
#                         , "Sites", "Issues", "Components"]]
search_df = results_df
df = pd.read_csv('./data/output_countries.csv')
df1 = pd.read_csv('./data/Issues_Keywords_Clusters.csv', encoding='latin-1')
component_df = pd.read_csv('./data/component_graph_data.csv')
issue_df = pd.read_csv('./data/issue_graph_data.csv')
clusterDesc = pd.read_csv('./data/manual_cluster_descriptions.csv')
clusters_df = pd.read_csv('./data/output_clusters_defined.csv', usecols = ['Response ID', 'manual_clusters'])


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
            title = 'Global Sentiment'),
      )]


layout = dict(
    title = 'This Week in Overall Global Sentiment of Mozilla Web Compat',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)


fig = dict(data=data, layout=layout)


# Hardcoded Fake Data
# arrayOfNames = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking']
# arrayOfNamesWords = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking', 'Words']
# arrayOfNamesDocs = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking', 'Docs']
# numClusters = 50
# traces = []
# clusterNames = list(df1)
# clusterNames.pop(0)
# print(clusterNames)
# df1 = df1.set_index('Issue')
# docs = df1.drop(arrayOfNamesWords, axis=0)
# words = df1.drop(arrayOfNamesDocs, axis=0)
# print(words.iloc[0].values[0])
# clusters = df1.drop(['Words', 'Docs'], axis=0)
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
num_days_range = 7  # week, set some way to toggle this instead of hardcoded
date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
date_filtered_df['Components'] = date_filtered_df['Components'].apply(
    lambda x: ast.literal_eval(x))  # gives warning but works, fix later
date_filtered_df['Issues'] = date_filtered_df['Issues'].apply(lambda x: ast.literal_eval(x))
# print(date_filtered_df['Issues'])
component_df = pd.Series([])
issue_df = pd.Series([])
comp_response_id_map = dict()
issue_response_id_map = dict()
for day in range(num_days_range):
    day_df = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]
    # count docs with components
    new_comp_info = day_df['Components'].apply(
        lambda x: pd.Series(x).value_counts()).sum()
    # count docs with no assigned components
    new_comp_info = pd.concat([new_comp_info,
                               date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
                                   lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])
    comp_response_id_map['Day ' + str(day + 1)] = dict()
    comps = new_comp_info.index.values
    for comp in comps:
        comp_response_id_map['Day ' + str(day + 1)][comp] = [];
    for index, row in day_df.iterrows():
        for comp in row['Components']:
            comp_response_id_map['Day ' + str(day + 1)][comp].append(
                row['Response ID'])  # TODO: can use map functions to make this faster
        if len(row['Components']) == 0 and 'No Label' in comps:
            comp_response_id_map['Day ' + str(day + 1)]['No Label'].append(row['Response ID'])
    component_df = pd.concat([component_df, new_comp_info.rename('Day ' + str(day + 1))], axis=1)
    new_issue_info = day_df['Issues'].apply(
        lambda x: pd.Series(x).value_counts()).sum()
    # count docs with no assigned components
    new_issue_info = pd.concat([new_issue_info,
                                date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
                                    lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])
    issue_response_id_map['Day ' + str(day + 1)] = dict()
    issues = new_issue_info.index.values
    for issue in issues:
        issue_response_id_map['Day ' + str(day + 1)][issue] = [];

    for index, row in day_df.iterrows():
        for issue in row['Issues']:
            issue_response_id_map['Day ' + str(day + 1)][issue].append(row['Response ID'])
        if len(row['Issues']) == 0 and 'No Label' in issues:
            issue_response_id_map['Day ' + str(day + 1)]['No Label'].append(row['Response ID'])
    issue_df = pd.concat([issue_df, new_issue_info.rename('Day ' + str(day + 1))], axis=1)


def initCompDF(results_df, num_days_range = 7):
    date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
    date_filtered_df['Components'] = date_filtered_df['Components'].apply(
        lambda x: ast.literal_eval(x))  # gives warning but works, fix later
    component_df = pd.Series([])
    for day in range(num_days_range):
        # count docs with components
        new_comp_info = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
            lambda x: pd.Series(x).value_counts()).sum()
        # count docs with no assigned components
        new_comp_info = pd.concat([new_comp_info,
                                   date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Components'].apply(
                                       lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])
        component_df = pd.concat([component_df, new_comp_info.rename('Day ' + str(day + 1))], axis=1)
    component_df = component_df.fillna(0).astype(int).drop(0, 1).rename_axis('Components')
    return component_df


def initIssueDF(results_df, num_days_range = 7):
    date_filtered_df = results2_df[results2_df['Day Difference'] <= num_days_range]
    date_filtered_df['Issues'] = date_filtered_df['Issues'].apply(lambda x: ast.literal_eval(x))
    issue_df = pd.Series([])
    for day in range(num_days_range):
        new_issue_info = date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
            lambda x: pd.Series(x).value_counts()).sum()
        # count docs with no assigned components
        new_issue_info = pd.concat([new_issue_info,
                                    date_filtered_df[date_filtered_df['Day Difference'] == day + 1]['Issues'].apply(
                                        lambda x: len(x)).value_counts().loc[[0]].rename({0: 'No Label'})])
        issue_df = pd.concat([issue_df, new_issue_info.rename('Day ' + str(day + 1))], axis=1)
    # Fill in component and issue df with 0 for Nan (?)
    issue_df = issue_df.fillna(0).astype(int).drop(0, 1).rename_axis('Issues')
    return issue_df


def updateComponentGraph(component_df):
    traces_component = []
    # Checking df for values:
    for index, row in component_df.iterrows():
        # print(list(row.keys()))
        traces_component.append(go.Bar(
            x=list(row.keys()),
            y=row.values,
            name=index,
            # hoverinfo='none',
            # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
            # customdata=docs.iloc[0].values
        ))
    # Stacked Bar Graph figure - components:
    layout_component = go.Layout(
        barmode='stack',
        title='Components Over Time',
        font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
        xaxis=dict(
            # showticklabels=False,
            title='Time'
        ),
        yaxis=dict(
            title='Count of Docs'
        )
    )
    fig_component = dict(data=traces_component, layout=layout_component)
    return fig_component


def updateIssuesGraph(issue_df):
    traces_issue = []
    # Checking df for values:
    for index, row in issue_df.iterrows():
        # print(list(row.keys()))
        traces_issue.append(go.Bar(
            x=list(row.keys()),
            y=row.values,
            name=index,
            # hoverinfo='none',
            # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
            # customdata=docs.iloc[0].values
        ))
    # Stacked Bar Graph figure - issues:
    layout_issue = go.Layout(
        barmode='stack',
        title='Issues Over Time',
        font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
        xaxis=dict(
            # showticklabels=True,
            title='Time'
        ),
        yaxis=dict(
            title='Count of Docs'
        )
    )
    fig_issue = dict(data=traces_issue, layout=layout_issue)
    return fig_issue


# CREATE FIRST TWO GRAPHS
component_df = initCompDF(results_df)
issue_df = initIssueDF(results_df)
fig_component = updateComponentGraph(component_df)
fig_issue = updateIssuesGraph(issue_df)


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


fig_comp_metrics = updateCompMetricsGraph()
fig_issue_metrics = updateIssuesMetricsGraph()

#prep data for displaying in stacked binary sentiment graph over time
#Grab unique dates from results_df
results_df["Date Submitted"] = pd.to_datetime(results_df["Date Submitted"])
unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()


# Page styling - sample:
PAGE_SIZE = 40
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)
# suppress exception of assigning callbacks to components that are genererated
# by other callbacks
app.config['suppress_callback_exceptions'] = True
app.title = 'Mozilla Analytics'
'''
Dash apps are composed of 2 parts. 1st part describes the app layout.
The 2nd part describes the interactivty of the app 
'''
tabs_styles = {
    'height': '44px',
    'width': '350px',
    'display': 'inline-block'
}
tab_style = {
    # 'borderBottom': '1px solid #d6d6d6',
    'padding': '11px',
    'backgroundColor': 'rgb(30,30,30)',
    'border': 'none',
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

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H1(
        children='Mozilla Customer Analytics',
        id="header",
    ),
    dcc.Tabs(id="tabs-styled-with-inline", children=[
        dcc.Tab(label='Overview', value='/overview', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Categories', value='/categories', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Sites', value='/sites', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Search', value='/search', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    print('current path', pathname)
    if pathname == '/categories':
        return categories_layout
    elif pathname == '/sites':
        return sites_layout
    elif pathname == '/search':
        return search_layout
    elif pathname == '/overview':
        return overview_layout
    else:
        return overview_layout


@app.callback(Output('url', 'pathname'),
              [Input('tabs-styled-with-inline', 'value')])
def update_url(tab): # bit of a hacky way of updating URL for now.
    return tab

list_page_children = []


overview_layout = html.Div([
            html.H2('Overview & Recent Trends'),
            dcc.Graph(id='graph', figure=fig),
            dcc.RadioItems(
                id='bin',
                options=[{'label': i, 'value': i} for i in [
                    'Yearly', 'Monthly', 'Weekly', 'Daily'
                ]],
                value='Daily',
                labelStyle={'display': 'inline'}
            ),
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
            ]),
            html.Div(id="bitch-div"),
            html.Div(id="bitch-div2")
        ])

@app.callback(
    Output('bitch-div2', 'children'),
    [Input('issue-graph', 'clickData')])
def display_click_data(clickData):
    if (len(clickData['points']) == 1):
        day = clickData['points'][0]['x']
        issue = clickData['points'][0]['customdata']
        ids = issue_response_id_map[day][issue]
        df = results_df[results_df['Response ID'].isin(ids)]
        return df
    else:
        return ''


@app.callback(
    Output('bitch-div', 'children'),
    [Input('comp-graph', 'clickData')])
def display_click_data(clickData):
    if (len(clickData['points']) == 1):
        day = clickData['points'][0]['x']
        component = clickData['points'][0]['customdata']
        ids = comp_response_id_map[day][component]
        df = results_df[results_df['Response ID'].isin(ids)]
        return df
    else:
        return ''
    # print(clickData['points'][0])
    # print(comp_response_id_map)
    # if (clickData):
    #     htmlArr = []
    #     data = clickData['points'][0]['customdata']
    #     docData = json.loads(json.dumps(ast.literal_eval(data)))
    #     for key, value in docData.items():
    #         docArray = []
    #         for doc in value:
    #             docArray.append(html.Div(doc, style={'outline': '1px dotted green'}))
    #         htmlArr.append(
    #             html.Div([
    #                 html.H4(key),
    #                 html.Div(children=docArray)
    #             ])
    #         )
    #     return htmlArr
    # return ''


@app.callback(Output('modal', 'style'), [Input('display_data','n_clicks_timestamp'),
                                         Input('close-modal', 'n_clicks_timestamp')])
def display_modal(openm, closem):
    if closem > openm:
        return {'display': 'none'}
    elif openm > closem:
        return {'display': 'block'}


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


@app.callback(
    Output('current-content', 'children'),
    [Input('trends-scatterplot', 'hoverData')])
def display_hover_data(hoverData):
    # get the row from the results
    try:
        r = results_df[results_df['Response ID'] == hoverData['points'][0]['customdata']]
        return html.H4(
            "The comment from {} is '{}'. The user was {}.".format(
                r.iloc[0]['Date Submitted'],
                r.iloc[0]['Feedback'],
                r.iloc[0]['Binary Sentiment']
            )
        )
    except TypeError:
        print('no hover data selected yet')


    # return ''


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


categories_layout = html.Div([
            html.Div(id = 'comp_slider_output'),
            dcc.Slider(id='comp_time_slider',
                       min=toggle_time_params['min'], max=toggle_time_params['max'],
                       step=toggle_time_params['step'], value=toggle_time_params['default'],
                       marks=toggle_time_params['marks']),
            dcc.Graph(id='graph2', figure=fig_component),

            html.Div(id='issue_slider_output'),
            dcc.Slider(id='issue_time_slider',
                       min=toggle_time_params['min'], max=toggle_time_params['max'],
                       step=toggle_time_params['step'], value=toggle_time_params['default'],
                       marks=toggle_time_params['marks']),
            dcc.Graph(id='graph3', figure=fig_issue),
            dcc.Graph(id='graph4', figure=fig_comp_metrics),
            dcc.Graph(id='graph5', figure=fig_issue_metrics),
            html.Div(className='row', children=[
                html.Div([
                    html.Div(id='click-data'),  # Doesn't do anything right now
                ]),
            ])
        ])

# Component DF Slider Callback
@app.callback(
    dash.dependencies.Output('comp_slider_output', 'children'),
    [dash.dependencies.Input('comp_time_slider', 'value')])
def update_output(value):
    return 'Past {} days of data'.format(value)


# Component DF Time Toggle Callback
@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('comp_time_slider', 'value')])
def update_output(value):
    component_df = initCompDF(results_df, value)
    fig_component = updateComponentGraph(component_df)
    return fig_component


divs = []


# Component DF Slider Callback
@app.callback(
    Output('click-data', 'children'),
    [Input('graph2', 'clickData')])
def display_click_data(clickData):
    if (clickData):
        htmlArr = []
        data = clickData['points'][0]['customdata']
        docData = json.loads(json.dumps(ast.literal_eval(data)))
        for key, value in docData.items():
            docArray = []
            for doc in value:
                docArray.append(html.Div(doc, style={'outline': '1px dotted green'}))
            htmlArr.append(
                html.Div([
                    html.H4(key),
                    html.Div(children=docArray)
                ])
            )
        return htmlArr
    return ''


@app.callback(
    dash.dependencies.Output('issue_slider_output', 'children'),
    [dash.dependencies.Input('issue_time_slider', 'value')])
def update_output(value):
    return 'Past {} days of data'.format(value)


# Component DF Time Toggle Callback
@app.callback(
    dash.dependencies.Output('graph3', 'figure'),
    [dash.dependencies.Input('issue_time_slider', 'value')])
def update_output(value):
    issue_df = initIssueDF(results_df, value)
    fig_issue = updateComponentGraph(issue_df)
    return fig_issue


common_df = results_df.groupby('Sites')['Sites'].agg(['count']).reset_index()

sites_layout = html.Div([
            html.H2('Mentioned Sites'),
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
            dcc.Graph(
                id='mentioned-site-graph',
                figure={
                    'data': [{
                        'x': common_df[common_df.columns[0]],
                        'y': common_df[common_df.columns[1]],
                        # 'customdata': results_df['Sites'].unique()[1:],
                        'type': 'bar'
                    }],
                    'layout': {
                        'title': "Feedback by Mentioned Site(s)",
                        'xaxis': {
                            'title': 'Mentioned Site(s)'
                        },
                        'yaxis': {
                            'title': 'Number of Feedback'
                        }
                    }
                }
            ),
            dt.DataTable(
                id='common-site-table',
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
        ])

@app.callback(
    dash.dependencies.Output('unique-site-count', 'children'),
    [dash.dependencies.Input('sites-date-range', 'start_date'),
     dash.dependencies.Input('sites-date-range', 'end_date')])
def update_site_count(start_date, end_date):    #update graph with values that are in the time range
    count = len(common_df.index)
    return 'There were {} unique site combinations mentioned in the user comments'.format(count)


@app.callback(
    dash.dependencies.Output('mentioned-site-graph', 'figure'),
    [dash.dependencies.Input('sites-date-range', 'start_date'),
     dash.dependencies.Input('sites-date-range', 'end_date')])
def update_graph_data(start_date, end_date):    #update graph with values that are in the time range
    print("swag")
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
#     Output('common-site-table', "data"),
#     [Input('common-site-table', "pagination_settings"),
#      Input('common-site-table', "sorting_settings"),
#      Input('mentioned-site-graph', "clickData")])
# def update_common_table(pagination_settings, sorting_settings, clickData):
#     # print(clickData)
#     dff = search_df[search_df['Sites'] == clickData['points'][0]['customdata']]
#     print('CLICKED DATA', clickData['points'][0]['customdata'])
#     if len(sorting_settings):
#         dff = dff.sort_values(
#             [col['column_id'] for col in sorting_settings],
#             ascending=[
#                 col['direction'] == 'asc'
#                 for col in sorting_settings
#             ],
#             inplace=False
#         )
#
#     return dff.iloc[
#            pagination_settings['current_page'] * pagination_settings['page_size']:
#            (pagination_settings['current_page'] + 1) * pagination_settings['page_size']
#            ].to_dict('rows')


search_layout = html.Div([
            html.H3('Search Raw Comments'),
            html.Label('Enter Search Term:'),
            dcc.Input(id='searchrequest', type='text', value='Type here'),
            dte.DataTable(  # Add fixed header row
                id='searchtable',
                rows=[{}],
                row_selectable=True,
                filterable=True,
                sortable=True,
                selected_row_indices=[],
            )
        ])


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
        fb = str(row['Feedback'])
        rv = str(request_value)
        isit = rv in fb
        if isit:
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), str(row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])]
            temp_df = pd.DataFrame([temp], columns=cnames)
            r_df = r_df.append(temp_df, ignore_index=True)
    return r_df.to_dict('rows')




if __name__ == '__main__':
    app.run_server(debug=True)

