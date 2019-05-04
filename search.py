import pandas as pd
from dash_interface import app, search_df
from dash.dependencies import Input, Output, State

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
            # temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), int(row['compound']),
            #         str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites'])]
            temp = [str(row['Response ID']), str(row['Date Submitted']), str(row['Country']), str("%.2f"%row['compound']),
                    str(row['Feedback']), str(row['Components']), str(row['Issues']), str(row['Sites']),
                    # str(row['User Agent'])
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
        # return 'display: none'


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
        print('reacheeeeeeed')
        return {'display': 'block'}
    else:
        return {'display': 'none'}