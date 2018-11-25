import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from dash.dependencies import Input, Output, State, Event


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

results_df = pd.read_csv("output.csv", encoding ="ISO-8859-1")

print (results_df.shape)
search_df = results_df[["Response ID", "Date Submitted", "Country","City"\
                        , "State/Region", "Binary Sentiment", "Positive Feedback"\
                        , "Negative Feedback", "Relevant Site", "compound", "neg", "neu", "pos"\
                        , "Sites", "Issues", "Components", "Processed Feedback"]]#print(df.columns)

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
        dcc.Tab(label='Trends', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Common Issues', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Search', value='tab-3', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline'),


    html.Div(children='Sentiment Breakdown using Dash/Plotly', style={
        'textAlign': 'center',
        'color': colors['text']
    })
])

@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Recent Trends'),
            dcc.Graph(
                id='graph-1-tabs',
                figure={
                    'data': [{
                        'x': results_df['Date Submitted'],
                        'y': results_df['compound'],
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
            html.Label('Here is a slider to vary # top sites to include'),
            dcc.Slider(id='hours', value=5, min=0, max=24, step=1)
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Common Issues'),
            dcc.Graph(
                id='graph-2-tabs',
                figure={
                    'data': [{
                        'x': [1, 2, 3],
                        'y': [5, 10, 6],
                        'type': 'bar'
                    }]
                }
            )
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Search Raw Comments'),
            html.Label('Enter Search Term:'),
            dcc.Input(value='Type here', type='text'),
            dash_table.DataTable( #add fixed header row
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
                    'minWidth': '50px', 'maxWidth': '200px',
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



if __name__ == '__main__':
    app.run_server(debug=True)

