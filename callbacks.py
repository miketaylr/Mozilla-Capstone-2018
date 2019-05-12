### Imports ###

from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import json
from collections import Counter
import urllib.parse
import dash_dangerously_set_inner_html

from constants import top_sites
from dash_interface import app, results2_df, search_df, toggle_time_params, sr_df, df_geo_sentiment, \
    updateGraph, updateGeoGraph, initCompDF, initIssueDF, component_df, issue_df, \
    runDrilldown, clusteringBarGraph
from layouts import main_layout, sites_layout, sentiment_layout, geoview_layout, components_layout, issues_layout, search_layout
import global_vars

### Note: the order the callbacks are placed in matters ###

### Global Callbacks ###

# Render Page Content
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname is None:
        return main_layout

    ids = []
    if 'sites' in pathname:
        ids = global_vars.global_site_modal_ids
    elif 'comp' in pathname:
        ids = global_vars.global_comp_modal_ids
    elif 'issues' in pathname:
        ids = global_vars.global_issue_modal_ids
    elif 'geo' in pathname:
        ids = global_vars.global_geo_modal_ids

    if 'classification' in pathname:
        # global_sites_list_df is the dataframe that contains the data that appears in the modal
        # ideally this should be fed through the same functions as results2_df to create the figures to display on the new page
        results_modal_df = results2_df[results2_df['Response ID'].isin(ids)]
        day_range_site_list = min(results_modal_df['Day Difference'].max(), toggle_time_params['max'])

        component_df_list, comp_response_id_map_list, comp_day_response_id_map_list = initCompDF(results_modal_df, day_range_site_list)
        list_component_df = component_df_list
        issue_df_list, issue_response_id_map_list, issue_day_response_id_map_list = initIssueDF(results_modal_df, day_range_site_list)
        list_issue_df = issue_df_list
        fig_component_list = updateGraph(component_df_list, 'Components Over Time', 7)
        fig_issue_list = updateGraph(issue_df_list, 'Issues Over Time', 7)
        return html.Div([
            html.Div([
                html.H1(
                    children='Categorization of Selected Data',
                    id="title",
                    style={'margin-left': '30px'}
                ),
            ]),
            html.Div([
                html.Div(id='list_comp_container',
                         className='one-half column list-slider-container',
                         children=[
                            html.H3('Components', className='page-title'),
                            html.Div(id='list_comp_slider_output'),
                            dcc.Slider(id='list_comp_time_slider',
                                        className='list-page-slider',
                                        min=toggle_time_params['min'], max=toggle_time_params['max'],
                                        step=toggle_time_params['step'], value=toggle_time_params['default'],
                                        marks=toggle_time_params['marks']),
                            dcc.Graph(id='list-comp-graph', figure=fig_component_list),
                         ]
                 ),
                html.Div(id='list_issue_container',
                         className='one-half column list-slider-container',
                         children=[
                            html.H3('Issues', className='page-title'),
                            html.Div(id='list_issue_slider_output'),
                            dcc.Slider(id='list_issue_time_slider',
                                        className='list-page-slider',
                                        min=toggle_time_params['min'], max=toggle_time_params['max'],
                                        step=toggle_time_params['step'], value=toggle_time_params['default'],
                                        marks=toggle_time_params['marks']),
                            dcc.Graph(id='list-issue-graph', figure=fig_issue_list),
                ]),
                ]),
            # html.Ul([html.Li(x) for x in global_top_selected_sites])
        ])
    elif 'clustering' in pathname:
        # global_sites_list_df is the dataframe that contains the data that appears in the modal
        # ideally this should be fed through the same functions as results2_df to create the figures to display on the new page
        results_modal_df = sr_df[sr_df['Response ID'].isin(ids)]
        print(results_modal_df)
        results = runDrilldown(results_modal_df)
        results = results.sort_values(by='Count', ascending=False)
        results = results.reset_index()
        pageChildren = []
        rootDict = dict()
        rootDict['name'] = ''
        rootDict['children'] = []
        for index, cluster in results.iterrows():
            responseIds = cluster['Response IDs']
            listArray = []
            for response in responseIds:
                feedback = results2_df[results2_df['Response ID'] == response]
                listArray.append(html.Li(feedback['Feedback']))
            dataDict = dict()
            wordArr = []
            topWords = cluster['Words'].split(',')
            if len(topWords) > 0:
                wordArr.append(topWords[0])
            if len(topWords) > 1:
                wordArr.append(topWords[1])
            dataDict['name'] = ",".join(wordArr)
            dataDict['value'] = len(listArray)
            clusterId = 'cluster_' + str(index+1)
            dataDict['id'] = clusterId
            rootDict['children'].append(dataDict)
            child=html.Details([
                html.Summary(('Cluster ' + str(index+1) + ' - ' + cluster['Words']), id=clusterId, className='clustering-summary-text'),
                html.P('Top Phrases: ' + cluster['Phrases'], className='clustering-top-text'),
                html.P('Feedback: ', className='clustering-top-text'),
                html.Ul(children=listArray, className='clustering-feedback'),
            ])
            pageChildren.append(child)

        results = results.transpose()
        fig = clusteringBarGraph(results, 'Clustering Analysis')
        return html.Div(className='clustering-page', children=[
            html.Div([
                html.H1(
                    children='Unsupervised Clustering of Selected Data',
                    id="title",
                    style={'margin-left': '30px'}
                ),
            ]),
            html.Div(id='div-d3', className="d3-container", **{'data-d3': json.dumps(rootDict)}, children=[
            dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
                <svg width="800" height="800"></svg>
            '''),
            ]),
            html.Div(children=pageChildren),
        ])
    else:
        return main_layout


@app.callback(Output('cluster-table', 'rows'),
              [Input('cluster-graph', 'clickData')])
def update_table(clickData):
    ids = clickData['points'][0]['customdata']
    dff = search_df[search_df['Response ID'].isin(ids)]
    cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
              'Feedback', 'Components', 'Issues', 'Sites']
    cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                 'Feedback', 'Components', 'Issues', 'Sites']
    dff = dff[cnames]
    dff.columns = cnamesnew
    return dff.to_dict('rows')


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

### Sentiment Page Callbacks ###

@app.callback(Output('binary-sentiment-ts', 'figure'),
              [Input('sentiment-frequency', 'value')])
def update_sentiment_graph(frequency):

    sad_df = (
        results_df[results_df["Binary Sentiment"] == "Sad"].groupby([pd.Grouper(key="Date Submitted", freq=frequency)]).count()
        .reset_index()
        .sort_values("Date Submitted")
    )

    happy_df = (
        results_df[results_df["Binary Sentiment"] == "Happy"].groupby([pd.Grouper(key="Date Submitted", freq=frequency)]).count()
        .reset_index()
        .sort_values("Date Submitted")
    )

    fig = {
        'data': [
            {
                'x': sad_df['Date Submitted'],
                'y': sad_df['Binary Sentiment'],
                'type': 'scatter',
                'name': "Sad"
            },
            {
                'x': happy_df['Date Submitted'],
                'y': happy_df['Binary Sentiment'],
                'type': 'scatter',
                'name': "Happy"
            }
        ],
        'layout': {
            'title': "Happy/Sad Breakdown",
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
    return fig

### Component Page Callbacks ###

@app.callback(
    Output('modal-comp', 'style'),
    [Input('close-comp-site', 'n_clicks'),
     Input('comp-graph', 'clickData')])
def display_comp_modal(closeClicks, clickData):
    if closeClicks > global_vars.compCloseCount:
        global_vars.compCloseCount = closeClicks
        return {'display': 'none'}
    elif clickData:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('modal-comp-table', 'rows'),
    [Input('comp-graph', 'clickData')])
def display_comp_click_data(clickData):
    #Set click data to whichever was clicked
    print('here in comp', clickData)
    if (clickData):
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            component = clickData['points'][0]['customdata']
            ids = global_vars.comp_response_id_map[day][component]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            ids = global_vars.comp_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global_vars.global_comp_modal_ids = list(dff['Response ID'])
        return dff.to_dict('rows')
    else:
        return []


@app.callback(
    Output('download-comp-link', 'href'),
    [Input('comp-graph', 'clickData')])
def update_comp_download_link(clickData):
    if clickData:
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            comp = clickData['points'][0]['customdata']
            ids = global_vars.comp_response_id_map[day][comp]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            ids = global_vars.comp_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ''


@app.callback(
    Output('comp_slider_output', 'children'),
    [Input('comp_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('comp-graph', 'figure'),
    [Input('comp_time_slider', 'value')])
def update_comp_graph_slider(value):
    fig_component = updateGraph(component_df, 'Components Over Time', value)
    return fig_component


@app.callback(
    Output('list_comp_slider_output', 'children'),
    [Input('list_comp_time_slider', 'value')])
def update_list_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('list-comp-graph', 'figure'),
    [Input('list_comp_time_slider', 'value')])
def update_list_comp_graph_slider(value):
    fig_component = updateGraph(list_component_df, 'Components Over Time', value)
    return fig_component


### Issue Page Callbacks ###

@app.callback(
    Output('modal-issue', 'style'),
    [Input('close-issue-site', 'n_clicks'),
     Input('issue-graph', 'clickData')])
def display_issue_modal(closeClicks, clickData):
    if closeClicks > global_vars.issueCloseCount:
        global_vars.issueCloseCount = closeClicks
        return {'display': 'none'}
    elif clickData:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('modal-issue-table', 'rows'),
    [Input('issue-graph', 'clickData')])
def display_issue_click_data(clickData):
    #Set click data to whichever was clicked
    print('here in issue', clickData)
    if (clickData):
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            issue = clickData['points'][0]['customdata']
            ids = global_vars.issue_response_id_map[day][issue]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            ids = global_vars.issue_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global_vars.global_issue_modal_ids = list(dff['Response ID'])
        return dff.to_dict('rows')
    else:
        return []


@app.callback(
    Output('download-issues-link', 'href'),
    [Input('issue-graph', 'clickData')])
def update_issue_download_link(clickData):
    if(clickData):
        if(len(clickData['points']) == 1):
            day = clickData['points'][0]['x']
            issue = clickData['points'][0]['customdata']
            ids = global_vars.issue_response_id_map[day][issue]
            dff = search_df[search_df['Response ID'].isin(ids)]
        else:
            day = clickData['points'][0]['x']
            ids = global_vars.issue_day_response_id_map[day]
            dff = search_df[search_df['Response ID'].isin(ids)]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ''


@app.callback(
    Output('issue_slider_output', 'children'),
    [Input('issue_time_slider', 'value')])
def update_issue_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('issue-graph', 'figure'),
    [Input('issue_time_slider', 'value')])
def update_issue_graph_slider(value):
    fig_issue = updateGraph(issue_df, 'Issues Over Time', value)
    return fig_issue


@app.callback(
    Output('list_issue_slider_output', 'children'),
    [Input('list_issue_time_slider', 'value')])
def update_list_issue_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('list-issue-graph', 'figure'),
    [Input('list_issue_time_slider', 'value')])
def update_list_issue_graph_slider(value):
    fig_issue = updateGraph(list_issue_df, 'Issues Over Time', value)
    return fig_issue

### Sites Page Callbacks ###

@app.callback(
    Output('top-view-selected', 'className'),
    [Input('top-sites-table', 'selected_row_indices')])
def disable_site_modal_button(selected_row_indices):
    if selected_row_indices:
        return 'view-selected-data'
    else:
        return 'view-selected-data-disabled'


@app.callback(
    Output('top-modal-site-table', 'rows'),
    [Input('top-view-selected', 'n_clicks')],
    [State('top-sites-table', 'rows'),
     State('top-sites-table', 'selected_row_indices')])
def update_site_modal_table(clicks, rows, selected_row_indices):
    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    if(clicks):
        dff = global_vars.global_filtered_top_sites_df[global_vars.global_filtered_top_sites_df['Sites'].isin(sites)]
        global_vars.global_site_modal_ids = list(dff['Response ID'])

        cnames = ['Feedback', 'Date Submitted', 'Country', 'compound',
                    'Components', 'Issues', 'Sites']
        cnamesnew = ['Feedback', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                   'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global_vars.global_top_selected_sites = sites
        return dff.to_dict('rows')
    return []


@app.callback(Output('top-modal-site', 'style'),
              [Input('top-close-modal-site', 'n_clicks'),
               Input('top-view-selected', 'n_clicks')],
              [State('top-sites-table', 'selected_row_indices')])
def display_modal(closeClicks, openClicks, selected_row_indices):
    if len(selected_row_indices) == 0:
        return {'display': 'none'}
    elif closeClicks > global_vars.topSiteCloseCount:
        global_vars.topSiteCloseCount = closeClicks
        return {'display': 'none'}
    elif openClicks:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(Output('top_sites_slider_output', 'children'),
    [Input('top_sites_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('top-sites-table', 'rows'),
    [Input('sites-date-range', 'start_date'),
     Input('sites-date-range', 'end_date')])
def update_comp_graph_slider(start_date, end_date):
    global results_df
    if(start_date is None or end_date is None):
        filtered_df = results_df
    else:
        filtered_df = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

    global_vars.global_filtered_top_sites_df = filtered_df

    sites_list = filtered_df['Sites'].apply(pd.Series).stack().reset_index(drop=True)
    sites_list = ','.join(sites_list).split(',')
    sites_list = [x for x in sites_list if '.' in x]

    sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
    sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
    sites_df = sites_df.sort_values(by=['Count'])
    sites_df = sites_df[sites_df['Site'] != 'None Found']

    top_sites_list = [item for item in sites_list if item in top_sites]
    top_sites_df = pd.DataFrame.from_dict(Counter(top_sites_list), orient='index').reset_index()
    top_sites_df = top_sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    top_sites_df = top_sites_df.sort_values(by=['Count'], ascending=False)
    return top_sites_df.to_dict('rows')


@app.callback(
    Output('other-view-selected', 'className'),
    [Input('other-sites-table', 'selected_row_indices')])
def disable_site_modal_button(selected_row_indices):
    if selected_row_indices:
        return 'view-selected-data'
    else:
        return 'view-selected-data-disabled'


@app.callback(
    Output('other-modal-site-table', 'rows'),
    [Input('other-view-selected', 'n_clicks')],
    [State('other-sites-table', 'rows'),
     State('other-sites-table', 'selected_row_indices')])
def update_site_modal_table(clicks, rows, selected_row_indices):
    print('here', clicks)

    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    print(sites)
    if(clicks):
        dff = global_vars.global_filtered_other_sites_df[global_vars.global_filtered_other_sites_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global_vars.global_site_modal_ids = list(dff['Response ID'])
        global_vars.global_other_selected_sites = sites
        return dff.to_dict('rows')
    return []


@app.callback(Output('other-modal-site', 'style'),
              [Input('other-close-modal-site', 'n_clicks'),
               Input('other-view-selected', 'n_clicks')],
              [State('other-sites-table', 'selected_row_indices')])
def display_modal(closeClicks, openClicks, selected_row_indices):
    if len(selected_row_indices) == 0:
        return {'display': 'none'}
    elif closeClicks > global_vars.otherSiteCloseCount:
        global_vars.otherSiteCloseCount = closeClicks
        return {'display': 'none'}
    elif openClicks:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('other_sites_slider_output', 'children'),
    [Input('other_sites_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('other-sites-table', 'rows'),
    [Input('sites-date-range', 'start_date'),
     Input('sites-date-range', 'end_date')])
def update_comp_graph_slider(start_date, end_date):
    global results_df
    if(start_date is None or end_date is None):
        filtered_df = results_df
    else:
        filtered_df = results_df[(results_df['Date Submitted'] > start_date) & (results_df['Date Submitted'] < end_date)]

    global_vars.global_filtered_other_sites_df = filtered_df

    sites_list = filtered_df['Sites'].apply(pd.Series).stack().reset_index(drop=True)
    sites_list = ','.join(sites_list).split(',')
    sites_list = [x for x in sites_list if '.' in x]

    sites_df = pd.DataFrame.from_dict(Counter(sites_list), orient='index').reset_index()
    sites_df = sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    sites_df['Formatted'] = sites_df['Site'].apply(lambda s: s.replace("https://", "").replace("http://", ""))
    sites_df = sites_df.sort_values(by=['Count'])
    sites_df = sites_df[sites_df['Site'] != 'None Found']

    other_sites_list = [item for item in sites_list if item not in top_sites and item != 'None Found']
    other_sites_df = pd.DataFrame.from_dict(Counter(other_sites_list), orient='index').reset_index()
    other_sites_df = other_sites_df.rename(columns={'index': 'Site', 0: 'Count'})
    other_sites_df = other_sites_df[other_sites_df['Count'] > 1]
    other_sites_df = other_sites_df.sort_values(by=['Count'], ascending=False)
    return other_sites_df.to_dict('rows')


@app.callback(
    Output('top-download-sites-link', 'href'),
    [Input('top-view-selected', 'n_clicks')],
    [State('top-sites-table', 'rows'),
     State('top-sites-table', 'selected_row_indices')])
def update_sites_download_link(clicks, rows, selected_row_indices):
    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    print(sites)
    if(clicks):

        dff = global_vars.global_filtered_top_sites_df[global_vars.global_filtered_top_sites_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    return ''


@app.callback(
    Output('other-download-sites-link', 'href'),
    [Input('other-view-selected', 'n_clicks')],
    [State('other-sites-table', 'rows'),
     State('other-sites-table', 'selected_row_indices')])
def update_sites_download_link(clicks, rows, selected_row_indices):
    table_df = pd.DataFrame(rows) #convert current rows into df

    if selected_row_indices:
        table_df = table_df.loc[selected_row_indices] #filter according to selected rows

    sites = table_df['Site'].values
    if(clicks):
        # ids = list(d['customdata'] for d in selectedData['points'])

        dff = global_vars.global_filtered_other_sites_df[global_vars.global_filtered_other_sites_df['Sites'].isin(sites)]
        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    return ''


@app.callback(
    Output('unique-site-count', 'children'),
    [Input('sites-date-range', 'start_date'),
     Input('sites-date-range', 'end_date')])
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
    no_sites_df = sites_df[sites_df['Site'] == 'None Found']
    sites_df = sites_df[sites_df['Site'] != 'None Found']
    count = len(sites_df.index)

    return html.Div([
        html.P(['Sites were mentioned {} times in the raw feedback. Unique sites were mentioned {} times.'.format(sites_df['Count'].sum(), count)]),
        html.P(['There were {} pieces of raw feedback that did not mention any sites.'.format(no_sites_df['Count'].sum())])
    ])

### World View Page Callbacks ###

@app.callback(
    Output('modal-geo-table', 'rows'),
    [Input('country-graph', 'clickData')])
def update_site_modal_table(clickData):
    if(clickData):
        selected_geo = clickData['points'][0]['text']
        dff = results_df[results_df['Country'] == selected_geo]
        global_vars.global_geo_modal_ids = list(dff['Response ID'])

        cnames = ['Feedback', 'Date Submitted', 'Country', 'compound',
                    'Components', 'Issues', 'Sites']
        cnamesnew = ['Feedback', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                   'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        dff.columns = cnamesnew
        global_vars.global_selected_geo = selected_geo
        return dff.to_dict('rows')
    return []


@app.callback(Output('modal-geo', 'style'),
              [Input('close-geo-modal', 'n_clicks'),
               Input('country-graph', 'clickData')])
def display_modal(closeClicks, openClick):
    if closeClicks > global_vars.geoCloseCount:
        global_vars.geoCloseCount= closeClicks
        return {'display': 'none'}
    elif openClick:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('download-geo-link', 'href'),
    [Input('country-graph', 'clickData')])
def update_comp_download_link(clickData):
    if clickData:
        selected_geo = clickData['points'][0]['text']
        dff = search_df[search_df['Country'] == selected_geo]

        cnames = ['Response ID', 'Date Submitted', 'Country', 'compound',
                  'Feedback', 'Components', 'Issues', 'Sites']
        cnamesnew = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
                  'Feedback', 'Components', 'Issues', 'Sites']
        dff = dff[cnames]
        csv_string = dff.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ''


@app.callback(
    Output('geo_time_slider', 'disabled'),
    [Input('geoview-radio', 'value')])
def abling_slider(radio_value):
    if radio_value == 'week':
        return False
    else:
        return True


@app.callback(
    Output('geo_slider_output', 'children'),
    [Input('geo_time_slider', 'value')])
def update_comp_output_slider(value):
    return 'Past {} days of data'.format(value)


@app.callback(
    Output('country-graph', 'figure'),
    [Input('geoview-radio', 'value'),
     Input('geo_time_slider', 'value')])
def update_geoview_graph(radio_value, slider_value):
    fig_geo = updateGeoGraph(df_geo_sentiment, radio_value, slider_value)
    return fig_geo

### Search Page Callbacks ###

@app.callback(
    Output('searchtable', 'rows'),
    [Input('searchrequest', 'n_submit')], # [Input('searchrequest', 'n_submit')],
    [State('searchrequest', 'value')])
def update_table(ns, request_value):
    df = search_df
    # cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
    #           'Feedback', 'Components', 'Issues', 'Sites']
    cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
              'Feedback', 'Components', 'Issues', 'Sites'
              # ,'Version'
              ]
    r_df = pd.DataFrame()
    # r_df = pd.DataFrame([cnames], columns=cnames)
    for index, row in df.iterrows():
        together = [str(row['Feedback']), str(row['Country']),
                    str(row['Components']), str(row['Issues']), str(row['Sites'])]
        fb = ''.join(together).lower()
        rv = str(request_value).lower()
        isit = rv in fb
        if isit:
            # vers = re.search(r'Firefox/\s*([\d.]+)', str(row['User Agent']))
            # temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), int(row['compound']),
            #         str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])]
            # print(vers.group())
            # table_vers=''
            # if vers is None:
            #     table_vers=''
            # else:
            #     table_vers = str(vers.group())
            print(row['compound'])
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), str("%.2f"%row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']),
                    str(row['Sites']),
                    # table_vers
                    ]
            temp_df = pd.DataFrame([temp], columns=cnames)
            r_df = r_df.append(temp_df, ignore_index=True)
    return r_df.to_dict('rows')


@app.callback(
    Output('search-download-link', 'href'),
    [Input('searchrequest', 'n_submit')],
    [State('searchrequest', 'value')])
def update_table(ns, request_value):
    df = search_df
    cnames = ['Response ID', 'Date Submitted', 'Country', 'Vader Sentiment Score',
              'Feedback', 'Components', 'Issues', 'Sites']
    r_df = pd.DataFrame()
    for index, row in df.iterrows():
        together = [str(row['Feedback']), str(row['Country']),
                    str(row['Components']), str(row['Issues']), str(row['Sites'])]
        fb = ''.join(together).lower()
        rv = str(request_value).lower()
        isit = rv in fb
        if isit:
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), str("%.2f"%row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])
                    ]
            temp_df = pd.DataFrame([temp], columns=cnames)
            r_df = r_df.append(temp_df, ignore_index=True)
    csv_string = r_df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


@app.callback(
    Output('search-count-reveal','children'),
    [Input('searchtable', 'rows')])
def set_search_count(dict_of_returned_df):
    if (len(dict_of_returned_df) > 0):
        if (len(dict_of_returned_df[0]) > 0):
            df_to_use = pd.DataFrame.from_dict(dict_of_returned_df)
            count = len(df_to_use.index)
            return u'Search returned {} results.'.format(count)
        else:
            return ''
    else:
        return u'Search returned no results.'

@app.callback(
    Output('search-loading', 'style'),
    [Input('search-table-container', 'style')],
    [State('searchrequest', 'value')])
def hide_loading(style, query):
    print(query)
    print(style['display'])
    if query and style['display'] == 'none':
        print('block')
        return {'display': 'block'}
    else:
        print('noneeeee')
        return {'display': 'none'}


@app.callback(
    Output('search-table-container','style'),
    [Input('search-count-reveal', 'children'),
     Input('searchtable', 'rows')])
def set_search_count(sentence, dict):
    if (len(dict[0]) > 0):
        return {'display': 'block',
                'animation': 'opac 1s'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('search-download-link','style'),
    [Input('search-count-reveal', 'children'),
     Input('searchtable', 'rows')])
def set_search_count(sentence, dict):
    if (len(dict[0]) > 0):
        return {'display': 'block'}
    else:
        return {'display': 'none'}