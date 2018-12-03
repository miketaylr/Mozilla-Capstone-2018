import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event
#import clustering as clustering
import ast
import json


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

results_df = pd.read_csv("./data/output_pipeline.csv", encoding ="ISO-8859-1")

print (results_df.shape) # SHOULD FILL NAN VALS AS WELL WHEN POSSIBLE
search_df = results_df[["Response ID", "Date Submitted", "Country","City"\
                        , "State/Region", "Binary Sentiment", "Feedback", "Relevant Site", "compound"\
                        , "Sites", "Issues", "Components"]]#print(df.columns)
#df = pd.read_csv('./data/output_countries.csv')
PAGE_SIZE = 40
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# suppress exception of assigning callbacks to components that are genererated
# by other callbacks
app.config['suppress_callback_exceptions'] = True

app.title = 'Mozilla Analytics'
'''
Dash apps are composed of 2 parts. 1st part describes the app layout.
The 2nd part describes the interactivty of the app 
'''

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(
        children='Mozilla Customer Feedback Analytics Tool',
        style={
            'textAlign': 'center',
            'color': 'orange'

        }
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
    })
])

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
           # dcc.Graph(id='graph', figure=fig),
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
           # dcc.Graph(id='graph2', figure=fig2),

            html.Div(className='row', children=[
                html.Div([
                    html.Div(id='click-data', target='_blank'),
                    # Above won't run on my pc for some reason unless I take out the target... -Carol
                    # html.Div(id='click-data'),
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

@app.callback(
    Output('current-content', 'children'),
    [Input('trends-scatterplot', 'hoverData')])
def display_hover_data(hoverData):
    # get the row from the results
    r = results_df[results_df['Response ID'] == hoverData['points'][0]['customdata']]

    return html.H4(
        "The comment from {} is '{}{}'. The user was {}.".format(
            r.iloc[0]['Date Submitted'],
            r.iloc[0]['Positive Feedback'] if r.iloc[0]['Positive Feedback'] != 'nan' else '',
            r.iloc[0]['Negative Feedback'] if r.iloc[0]['Negative Feedback'] != 'nan' else '',
            r.iloc[0]['Binary Sentiment']
        )
    )

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


divs = []
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


if __name__ == '__main__':
    app.run_server(debug=True)

