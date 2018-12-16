import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
import clustering as clustering
import ast
import json
from datetime import datetime as datetime
from constants import WORDS_TO_COMPONENT, WORDS_TO_ISSUE


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# Reading in data:
results_df = pd.read_csv("./data/output_pipeline.csv", encoding ="ISO-8859-1")
print (results_df.shape) # SHOULD FILL NAN VALS AS WELL WHEN POSSIBLE
# search_df = results_df[["Response ID", "Date Submitted", "Country","City"\
#                         , "State/Region", "Binary Sentiment", "Positive Feedback"\
#                         , "Negative Feedback", "Relevant Site", "compound"\
#                         , "Sites", "Issues", "Components"]]#, "neg", "neu", "pos" "Processed Feedback" print(df.columns)
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
      ) ]


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
results_df['Day Difference'] = (reference - pd.to_datetime(results_df['Date Submitted'],
                                                           format='%Y-%m-%d %H:%M:%S')).dt.days + 1

def initCompDF(results_df, num_days_range = 7):
    date_filtered_df = results_df[results_df['Day Difference'] <= num_days_range]
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
    date_filtered_df = results_df[results_df['Day Difference'] <= num_days_range]
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
        print(list(row.keys()))
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


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


list_page_children = []


main_layout = html.Div(children=[
    html.H1(
        children='Mozilla Customer Analytics',
        id="header",
    ),
    dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
        dcc.Tab(label='Overview', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Categories', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Sites', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Search', value='tab-4', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline'),


    html.Div(children='Sentiment Breakdown using Dash/Plotly', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Div(id='hidden-div', style={'display': 'none'})
])

list_layout = html.Div(children=[
    html.H1(
        children='List',
        style={
            'color': 'orange'

        }
    ),
    html.Div(children=list_page_children)
    
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/list':
        return main_layout
    elif pathname == '/page-2':
        return main_layout
    else:
        return main_layout  

#prep data for displaying in stacked binary sentiment graph over time
#Grab unique dates from results_df
results_df["Date Submitted"] = pd.to_datetime(results_df["Date Submitted"])
unique_dates = results_df["Date Submitted"].map(pd.Timestamp.date).unique()
common_df = test2 = results_df.groupby('Sites')['Sites'].agg(['count']).reset_index()

@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Overview & Recent Trends'),
            dcc.Graph(id='graph', figure=fig),
            dcc.RadioItems(
                id='bin',
                options=[{'label': i, 'value': i} for i in [
                    'Yearly', 'Monthly', 'Weekly', 'Daily'
                ]],
                value='Daily',
                labelStyle={'display': 'inline'}
            ),
            dcc.Graph(
                id='binary-sentiment-ts',
                figure={
                    'data': [{
                        'x': unique_dates,
                        'y': results_df[results_df["Binary Sentiment"] == "Happy"].groupby([results_df['Date Submitted'].dt.date])['Binary Sentiment'].count().values,
                        'type': 'bar',
                        'name': "Happy"
                        }, {
                        'x': unique_dates,
                        'y': results_df[results_df["Binary Sentiment"] == "Sad"].groupby([results_df['Date Submitted'].dt.date])['Binary Sentiment'].count().values,
                        'type': 'bar',
                        'name': "Sad"
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
            ),
            dcc.Graph(
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
            ),
            html.Div([
                html.Div(
                    className='six columns',
                    children=dcc.Graph(id='trend-data-histogram')
                ),
                html.Div(
                    className='six columns',
                    id='current-content'
                )
            ])
            # html.Label('Here is a slider to vary # top sites to include'),
            # dcc.Slider(id='hours', value=5, min=0, max=24, step=1)
        ])
    elif tab == 'tab-2':
        return html.Div([
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
                    html.Div(id='click-data'),
                ]),
            ])
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Sites'),
            dcc.Graph(
                id='mentioned-site-graph',
                figure={
                    'data': [{
                        'x': common_df[common_df.columns[0]],
                        'y': common_df[common_df.columns[1]],
                        'customdata': results_df['Sites'].unique()[1:],
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
                        'if': {'column_id': 'Positive Feedback'},
                        'textAlign': 'left'
                    },
                    {
                        'if': {'column_id': 'Negative Feedback'},
                        'textAlign': 'left'
                    },

                ],
                css=[{
                    'selector': '.dash-cell div.dash-cell-value',
                    'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',

                }],
            ),
            html.H4('Similar graphs & reactive table for issue/feature categories')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Search Raw Comments'),
            html.Label('Enter Search Term:'),
            dcc.Input(value='Type here', type='text'),
            dt.DataTable( #add fixed header row
                id='search-table',
                columns=[{"name": i, "id": i} for i in search_df.columns],
                pagination_settings={
                    'current_page': 0,
                    'page_size': PAGE_SIZE
                },
                pagination_mode='be',
                sorting='be',
                sorting_type='single',
                sorting_settings=[],
                filtering='be',
                filtering_settings='',
                data=search_df.head(50).to_dict("rows"),
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
                        'if': {'column_id': 'Positive Feedback'},
                        'textAlign': 'left'
                    },
                    {
                        'if': {'column_id': 'Negative Feedback'},
                        'textAlign': 'left'
                    },

                ],
                css=[{
                        'selector': '.dash-cell div.dash-cell-value',
                        'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',

                     }],
            )
        ])

# @app.callback(
#     Output('current-content', 'children'),
#     [Input('trends-scatterplot', 'hoverData')])
# def display_hover_data(hoverData):
#     # get the row from the results
#     r = results_df[results_df['Response ID'] == hoverData['points'][0]['customdata']]

#     return html.H4(
#         "The comment from {} is '{}{}'. The user was {}.".format(
#             r.iloc[0]['Date Submitted'],
#             r.iloc[0]['Positive Feedback'] if r.iloc[0]['Positive Feedback'] != 'nan' else '',
#             r.iloc[0]['Negative Feedback'] if r.iloc[0]['Negative Feedback'] != 'nan' else '',
#             r.iloc[0]['Binary Sentiment']
#         )
#     )

@app.callback(
    Output('trend-data-histogram', 'figure'),
    [Input('trends-scatterplot', 'selectedData')])
def display_selected_trend_data(selectedData):
    #return table matching the current selection

    ids = list(d['customdata'] for d in selectedData['points'])
    df = search_df[search_df['Response ID'].isin(ids)]
    print(ids)
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
    Output('search-table', "data"),
    [Input('search-table', "pagination_settings"),
     Input('search-table', "sorting_settings"),
     Input('search-table', "filtering_settings")])
def update_table(pagination_settings, sorting_settings, filtering_settings):
    print(sorting_settings)
    print(filtering_settings)
    filtering_expressions = filtering_settings.split(' && ')
    dff = search_df

    for filter in filtering_expressions:
        if ' eq ' in filter:
            col_name = filter.split(' eq ')[0]
            filter_value = filter.split(' eq ')[1]
            dff = dff.loc[dff[col_name] == filter_value]
        if ' > ' in filter:
            col_name = filter.split(' > ')[0]
            filter_value = float(filter.split(' > ')[1])
            dff = dff.loc[dff[col_name] > filter_value]
        if ' < ' in filter:
            col_name = filter.split(' < ')[0]
            filter_value = float(filter.split(' < ')[1])
            dff = dff.loc[dff[col_name] < filter_value]

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
    Output('common-site-table', "data"),
    [Input('common-site-table', "pagination_settings"),
     Input('common-site-table', "sorting_settings"),
     Input('mentioned-site-graph', "clickData")])
def update_common_table(pagination_settings, sorting_settings, clickData):
    dff = search_df[search_df['Sites'] == clickData['points'][0]['customdata']]
    print('CLICKED DATA', clickData['points'][0]['customdata'])
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

# Component DF Slider Callback
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

if __name__ == '__main__':
    app.run_server(debug=False)

