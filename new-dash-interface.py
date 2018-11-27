import dash
import json
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import clustering as clustering
import ast

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv('./data/output_countries.csv')
df1 = pd.read_csv('./data/Issues_Keywords_Clusters.csv', encoding='latin-1')
clusterDesc = pd.read_csv('./data/manual_cluster_descriptions.csv')

data = [dict(
    type='choropleth',
    locations=df['CODE'],
    z=df['Sentiment'],
    text=df['COUNTRY'],
    colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"], \
                [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]],
    autocolorscale=False,
    reversescale=True,
    marker=dict(
        line=dict(
            color='rgb(180,180,180)',
            width=0.7
        )),
    colorbar=dict(
        autotick=False,
        tickprefix='',
        title='Global Sentiment'),
)]

layout = dict(
    title='This Week in Overall Global Sentiment of Mozilla Web Compat',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection=dict(
            type='Mercator'
        )
    )
)

fig = dict(data=data, layout=layout)

arrayOfNames = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug',
                'Media Playback', 'Security', 'Search Hijacking']
arrayOfNamesWords = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug',
                     'Media Playback', 'Security', 'Search Hijacking', 'Words']
arrayOfNamesDocs = ['Performance', 'Crashes', 'Layout Bugs', 'Regressions', 'Not Supported', 'Generic Bug',
                    'Media Playback', 'Security', 'Search Hijacking', 'Docs']
numClusters = 50
traces = []

# Hardcoded Fake Data
# clusterNames = list(df1)
# clusterNames.pop(0)
# print(clusterNames)
# df1 = df1.set_index('Issue')
# docs = df1.drop(arrayOfNamesWords, axis=0)
# words = df1.drop(arrayOfNamesDocs, axis=0)
# print(words.iloc[0].values[0])
# clusters = df1.drop(['Words', 'Docs'], axis=0)

# Dynamic Data
df2 = clustering.runVis(numClusters)
categoryDict = pd.Series(clusterDesc.description.values, index=clusterDesc.clusters_types).to_dict()

docs = df2.tail(1)
df2 = df2[:-1]
phrases = df2.tail(1)
df2 = df2[:-1]
words = df2.tail(1)
df2 = df2[:-1]
clusters = df2
clusters = clusters.rename(index=categoryDict)


def update_point(trace):
    print(trace)
    return


for index, row in clusters.iterrows():
    row = list(row)
    traces.append(go.Bar(
        x=words.iloc[0].values,
        y=row,
        name=index,
        hoverinfo='x+y+name',
        # customdata=str(phrases.iloc[0].values + '&&' + docs.iloc[0].values)
        customdata=docs.iloc[0].values

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

fig2 = dict(data=traces, layout=layout2)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
'''
Dash apps are composed of 2 parts. 1st part describes the app layout.
The 2nd part describes the interactivty of the app 
'''
colors = {
    'background': '#FFFFFF',
    'text': '#7FDBFF'
}

app.layout = html.Div(id='test', style={'backgroundColor': colors['background']}, children=[
    dcc.Graph(id='graph', figure=fig),
    dcc.Graph(id='graph2', figure=fig2),

    html.Div(className='row', children=[
        html.Div([
            html.Div(id='click-data', target='_blank'),
        ]),
    ])
])

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
    app.run_server(debug=False)