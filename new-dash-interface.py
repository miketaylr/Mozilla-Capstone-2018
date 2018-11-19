import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_csv('./data/output.csv')

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

trace1 = go.Bar(
    x=['Youtube', 'Baidu', 'Facebook', 'Yahoo', 'Vimeo'],
    y=[20, 14, 23, 16, 45],
    name='Performance'
)
trace2 = go.Bar(
    x=['Youtube', 'Baidu', 'Facebook', 'Yahoo', 'Vimeo'],
    y=[0, 18, 29, 3, 11],
    name='Crashes'
)
trace3 = go.Bar(
    x=['Youtube', 'Baidu', 'Facebook', 'Yahoo', 'Vimeo'],
    y=[1, 5, 27, 19, 22],
    name='Security'
)
trace4 = go.Bar(
    x=['Youtube', 'Baidu', 'Facebook', 'Yahoo', 'Vimeo'],
    y=[12, 18, 2, 0, 1],
    name='Not supported'
)
trace5 = go.Bar(
    x=['Youtube', 'Baidu', 'Facebook', 'Yahoo', 'Vimeo'],
    y=[19, 0, 4, 16, 8],
    name='Regressions'
)

data2 = [trace1, trace2, trace3, trace4, trace5]
layout2 = go.Layout(
    barmode='stack'
)


fig2 = dict( data=data2, layout=layout2)


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