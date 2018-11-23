import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt 
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State, Event


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

results_df = pd.read_csv("./data/output.csv", encoding ="ISO-8859-1")

print (results_df.shape)
#print(df.columns)

df = pd.read_csv('./data/output_countries.csv')
df1 = pd.read_csv('./data/Issues_Keywords_Clusters.csv')

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

fig = dict( data=data, layout=layout)

arrayOfNames = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug', 'Media Playback', 'Security', 'Search Hijacking']
numClusters = 5
traces = []

clusterNames = list(df1)
clusterNames.pop(0)
print(clusterNames)
df1 = df1.set_index('Issue')
words = df1.drop(arrayOfNames, axis=0)
print(words.iloc[0].values[0])

clusters = df1.drop('Words', axis=0)

print(clusters)

for index, row in clusters.iterrows():
    row = list(row)
    print(index)
    traces.append(go.Bar(
        x = words.iloc[0].values,
        y = row,
        name=index,
        hoverinfo='x+y+name',
    ))


layout2 = go.Layout(
    barmode='stack',
    title='Issue Clusters',
    font=dict(family='Arial Bold', size=18, color='#7f7f7f'),
    xaxis=dict(
        showticklabels=False,
        title='Clusters'
    ),
    yaxis=dict(
        title='Count of Issues'
    )
)


fig2 = dict( data=traces, layout=layout2)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
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
        dcc.Tab(label='Trends', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Common Issues', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Search', value='tab-3', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline'),


    html.Div(children='Sentiment Breakdown using Dash/Plotly', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'Happy'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Sad'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
])

@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Recent Trends'),
            # dcc.Graph(
            #     id='graph-1-tabs',
            #     figure={
            #         'data': [{
            #             'x': [1, 2, 3],
            #             'y': [3, 1, 2],
            #             'type': 'bar'
            #         }]
            #     }
            # )
            dcc.Graph(id='graph1', figure=fig)
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Common Issues'),
            # dcc.Graph(
            #     id='graph-2-tabs',
            #     figure={
            #         'data': [{
            #             'x': [1, 2, 3],
            #             'y': [5, 10, 6],
            #             'type': 'bar'
            #         }]
            #     }
            # )
            dcc.Graph(id='graph2', figure=fig2)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Search Raw Comments'),
            html.Label('Enter Search Term:'),
            dcc.Input(value='Type here', type='text'),
            dcc.Graph(
                id='graph-2-tabs',
                figure={
                    'data': [{
                        'x': [1, 2, 3],
                        'y': [5, 10, 6],
                        'type': 'bar'
                    }]
                }
            ),
            dt.DataTable(
                    rows=results_df,
                    columns=results_df.columns,
                    filterable=True,
                    id='datatable')
        ])

if __name__ == '__main__':
    app.run_server(debug=True)