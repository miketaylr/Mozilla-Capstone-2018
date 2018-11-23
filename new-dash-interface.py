import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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
'''
Dash apps are composed of 2 parts. 1st part describes the app layout.
The 2nd part describes the interactivty of the app 
'''
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    dcc.Graph(id='graph', figure=fig),
    dcc.Graph(id='graph2', figure=fig2),
])

if __name__ == '__main__':
    app.run_server(debug=True)