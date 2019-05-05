import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table_experiments as dte

from dash_interface import results_df, top_sites_df, other_sites_df, unique_dates, search_df, toggle_time_params, \
    fig_geo, fig_component, fig_issue

PAGE_SIZE = 40

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
    ]),
    html.Div(
        className='sites-table-container',
        children=[
            html.Div(id='top-sites-table-container', className='sites-table',
                children=[
                    html.H4('Alexa Top 100 Sites', className='page-title'),
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
    html.H3('How Do Users Feel About Firefox?', className='page-title'),
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
                            'family': 'Montserrat, Helvetica Neue, Helvetica, sans-serif',
                            'color': 'white',
                        },
                        'xaxis': {
                            'title': 'Time'
                        },
                        'yaxis': {
                            'title': 'Amount of Feedback'
                        },
                        'font': {
                            'family': 'Montserrat, Helvetica Neue, Helvetica, sans-serif',
                            'size': 12,
                            'color': 'white',
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
    html.H3('How Does The World Feel About Firefox?', className='page-title'),
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
            {'label': 'Average country sentiment over selected time range', 'value': 'week'},
            {'label': 'How different is the country sentiment now compared to usual?', 'value': 'norm'},
            {'label': 'In general, how different is the country sentiment from all other countries?', 'value': 'globalNorm'},
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
    html.H3('Where Did Firefox Break?', className='page-title'),
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
    html.H3('What Are The Firefox Browser Issues?', className='page-title'),
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
    style={'text-align': 'center'}
)