#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:35:09 2022

@author: kamila
"""
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, no_update, ctx
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import geopandas as gpd
import pandas as pd
import plotly.io as pio
import json
from datetime import datetime
import numpy as np
import base64
import io
import ast
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1300)

# plot postcodes
pio.renderers.default='browser'


# import df
pc_sales_df = gpd.read_file('pc_sales_df.geojson')
pc_sales_df.head(3)

# join ship from locations
delivery_loc = pd.read_csv('pc_shipfrom_loc.csv')
delivery_loc['Row Labels'] = delivery_loc['Row Labels'].astype('str')
delivery_loc['Row Labels'] = delivery_loc['Row Labels'].apply(lambda x: '0' + x if x.startswith('8') else x)
delivery_loc.replace(0, False, inplace = True)
delivery_loc.replace(1, True, inplace = True)

pc_sales_df = pc_sales_df.merge(delivery_loc, left_on = 'poa_code21', right_on = 'Row Labels', how = 'left')


# ready geometries

pc_sales_df['geometry'] = pc_sales_df.to_crs(pc_sales_df.estimate_utm_crs()).simplify(2000).to_crs(pc_sales_df.crs) # simplify boundaries to 1km
pc_sales_df = pc_sales_df.to_crs( epsg = 4326) # change to lat/long
pc_sales_df.set_index('poa_code21', inplace = True)

df = pc_sales_df
df['terr_colour'] = np.NaN



# filter by state to speed up plotting


# state_df = pc_sales_df[(pc_sales_df.codestte == state) | (pc_sales_df[state] == True)]
# state_df = pc_sales_df[pc_sales_df.codestte == state]
state_coords = {'NT' : [-19.491411, 132.550964],
                'NSW' : [-33.872762037132375, 147.22963557432993],
                'VIC' : [-37.020100, 144.964600],
                'QLD' : [-20.917574, 142.702789],
                'SA' : [-30.000233, 136.209152],
                'WA' : [-25.953512, 117.857048],
                'TAS' : [-41.640079, 146.315918]
                }





# initialise variables
df_state = df[df.codestte == 'NSW']

postcode_colour_d = dict()

fig = px.choropleth_mapbox(df_state,
                           geojson = df_state.geometry,
                           locations = df_state.index,
                           opacity = 0.2,
                           center = {'lat' : -33.872762037132375, 'lon' : 147.22963557432993},
                           zoom = 4.5,
                           height = 1200,
                           width = 1600,
                           mapbox_style="carto-positron",
                           # uirevision = 'Retain user zoom preferences'
)

# app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

app.layout = html.Div(
    [
     dbc.Row(                     
         dcc.Dropdown(id = 'select_state',
                      options = [{'label' : i, 'value' : i} for i in pc_sales_df.codestte.unique()],
                      value = 'NSW',
                      multi = True,
                      style = {'width' : 400}  
        )
    ),
     
     dbc.Row(
         [
             dbc.Col(
                 [                                
                    dcc.Graph(id = 'graph', figure = fig)                
                 ], width = 9       
         
            ),
             
             dbc.Col(
                 [  
                     html.Br(),
                     html.Br(),
                     html.Div(
                         [
                             html.Button('Download map and legend', id = 'btn_download_map', style = {'width' : 200, 'height' : 75}),
                     
                             dcc.Upload('Import data', id = 'import_data',
                                style = {#'width' : '30%', 
                                        #'height' : '80px',
                                        'lineheight' : '200px',
                                        'borderWidth' : '1px',
                                        'textAlign' : 'center',
                                        'margin' : '20px',
                                        'cursor' : 'pointer',
                                        'background-color' : 'rgb(201,201,201)'
                                        }
                            ),
                     
                            html.Button('Add new legend entry', id = 'btn_add_legend_entry', style = {'width' : 200, 'height' : 75})
                         ], style = {'display' : 'flex'}
                    ),
                     
                    #  html.Div(id = {'type' : 'legend_div',
                    #                 'index': 0},
                    #           children =
                    #      [                               
                    #          dbc.Input(id = {'type' : 'colour_input',
                    #                          'index' : 0},
                    #                    type = 'color',
                    #                    style = {'width' : 75, 'height' : 50}                             
                    #         ),
                             
                    #          dbc.Input(id = {'type' : 'colour_label',
                    #                          'index' : 0},
                    #                    type = 'text',
                    #                    style = {'width' : 400, 'height' : 50}
                    #         )
                    #     ], style = {'display' : 'flex'}
                    # ),
                    
                    html.Div(id = 'test_area'),
                    
                    dcc.Store(id = 'legend_triggered'), # which legend was last triggered
                    
                    dcc.Store(id = 'map_legend_data'), # map data stored for download
                    
                    dcc.Store(id = 'map_upload'),
                    
                    dcc.Store(id = 'nn_click_count', data = 0),
                    
                    dcc.Store(id = 'legend_data'),
                                   
                    dcc.Download(id = 'download_map_data'),
                    
                    html.Button('test button', id = 'test_button', style = {'width' : 200, 'height' : 75}),
                    
                    html.Div(id = 'legend_container', children = []),

                    html.Div(id = 'test_import_area'),
                    
                    html.Div(id = 'legend_label_test_area', children = [])

                           
                 ]
            )                 
        ]
    )
    ]
)
# access legend labels, colours
@app.callback(
    Output('legend_data', 'data'),
    Output('legend_label_test_area', 'children'),
    Input({'type' : 'legend_div', 'index' : ALL},'n_clicks'),
    State({'type' : 'colour_label', 'index' : ALL}, 'value'),
    State({'type' : 'colour_input', 'index' : ALL}, 'value'),
    prevent_initial_call = True
    )
def get_legend_label(n_clicks, legend_labels, legend_colour):
    legend_dict = dict(zip(legend_colour, legend_labels))
    return legend_dict , html.Pre('get_legend_label function output: ' + str(legend_dict))


# import data
@app.callback(
    Output('map_upload', 'data'),
    Output('test_import_area', 'children'), # populate div while debugging
    Input('import_data', 'contents'),
    prevent_initial_call = True
)
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded_content = base64.b64decode(content_string)
    decoded_str = decoded_content.decode('utf-8')
    return decoded_str, html.Pre('output from parse_contents function: ' + decoded_str[:50])


# download data
@app.callback(
    Output('download_map_data', 'data'),
    # Output('legend_label_test_area', 'children'),
    Input('btn_download_map', 'n_clicks'),
    State('map_legend_data', 'data'),
    State('legend_data', 'data'),
    prevent_initial_call = True
)

def download_data(n_clicks, data, legend_data): # this version is outputting map and legend data as list of dictionaries
    if n_clicks is not None:
        download_legend_data = json.dumps([legend_data, data])
        download_data = json.dumps(data)
        content_dict = {'content' : download_legend_data}
    return dict(content_dict, filename = 'mapdata.text') #, html.Pre(str(download_legend_data))
    


# add new legend entry
@app.callback(
    Output('legend_container', 'children'),
    # Output('btn_add_legend_entry', 'n_clicks'),
    Output('nn_click_count', 'data'),
    Input('btn_add_legend_entry', 'n_clicks'),
    Input('map_upload', 'data'),
    State('legend_container', 'children'),
    State('nn_click_count', 'data'),
    prevent_initial_call = True
)
def add_legend(n_clicks, map_upload, container, nn_clicks):
    if map_upload is None:
        if n_clicks is None:
            return container, nn_clicks
        new_legend = html.Div(id = {'type' : 'legend_div',
                                    'index': nn_clicks},
                              children =
                                [                               
                                         dbc.Input(id =  {'type' : 'colour_input',
                                                          'index' : nn_clicks},
                                                   type = 'color',
                                                   style = {'width' : 75, 'height' : 50}
                                         
                                        ),
                                         
                                         dbc.Input(id = {'type' : 'colour_label',
                                                          'index' : nn_clicks},
                                                   type = 'text',
                                                   style = {'width' : 400, 'height' : 50}
                                        )
                                ], style = {'display' : 'flex'}
                    ) 
        nn_clicks += 1
        container.append(new_legend)
        
        return container, nn_clicks
    
    else:
        json_list = json.loads(map_upload)
        pc_colour_d = json_list[0]
        for col, label in pc_colour_d.items():
            new_legend = html.Div(id = {'type' : 'legend_div',
                                        'index': nn_clicks},
                                  children =
                            [                               
                                     dbc.Input(id =  {'type' : 'colour_input',
                                                      'index' : nn_clicks},
                                               value = str(col),
                                               type = 'color',
                                               style = {'width' : 75, 'height' : 50}
                                     
                                    ),
                                     
                                     dbc.Input(id = {'type' : 'colour_label',
                                                      'index' : nn_clicks},
                                               type = 'text',
                                               value = label,
                                               style = {'width' : 400, 'height' : 50}
                                    )
                            ], style = {'display' : 'flex'}
                        ) 
            container.append(new_legend)
            nn_clicks  += 1
        return container, nn_clicks
    
# colour selected listener
@app.callback(
    Output('legend_triggered', 'data'),
    # Output('test_area', 'children'),
    Input({'type' : 'legend_div', 'index' : ALL}, 'n_clicks'),
    prevent_initial_call = True
    ) 
def colour_listener(colourLastSelected):
    return ctx.triggered_id.index #, html.Pre(ctx.triggered_id.index) 
    

# update map
@app.callback(
    Output('graph', 'figure'),
    Output('graph', 'clickData'),
    Output('map_legend_data', 'data'),
    Output('test_area', 'children'),
    Output('map_upload', 'clear_data'),
    Input('graph', 'clickData'),
    Input('select_state', 'value'),
    State({'type' : 'colour_input', 'index' : ALL}, 'value'),  
    State('legend_triggered', 'data'),
    Input('map_upload', 'data'),
    prevent_initial_call = True
    )
def update_map(selectPC, selectedState, selectedColor, legendTriggered, mapUpload):
    clear_map_data = False
    if mapUpload is not None:
        json_str_list = json.loads(mapUpload)
        json_str = json_str_list[1]
        print('json_str: ' + str(json_str))
        postcode_colour_d.update(json_str)
        clear_map_data = True
   
    if selectPC is not None:
        postcode = selectPC['points'][0]['location']
        
        if postcode in postcode_colour_d:
            postcode_colour_d.pop(postcode)
        else:   
            triggeredColor = selectedColor[legendTriggered]
            postcode_colour_d[postcode] = triggeredColor
    
    postcode_list = list(postcode_colour_d.keys())
    print('postcode_list: ' + str(postcode_list))
    
    if type(selectedState) == str:
        df_state = df[df.codestte == selectedState]
        
    else: 
        df_state = df[df.codestte.isin(selectedState)]

    fig = px.choropleth_mapbox(df_state,
                                geojson = df_state.geometry,
                                locations = df_state.index,
                                opacity = 0.2,
                                #center = {'lat' : state_coords[selectedState][0], 'lon' : state_coords[selectedState][1]},
                                center = {'lat' : -33.872762037132375, 'lon' : 147.22963557432993},
                                zoom = 4.5,
                                height = 1200,
                                width = 1600,
                                mapbox_style="carto-positron"
    )
    
    selected_df = df_state.loc[postcode_list, :]
    selected_df.reset_index(inplace = True)
    selected_df['terr_colour'] = selected_df['poa_code21'].apply(lambda x: postcode_colour_d.get(x))
    selected_df.set_index('poa_code21', inplace = True)
    print('selected_df: ' + str(selected_df))
    for colour in selected_df.terr_colour.unique():
        dff = selected_df[selected_df.terr_colour == colour]
        print('df used to create new trace: ' + str(dff))
        fig.add_trace(
            px.choropleth_mapbox(dff,
                                  geojson = dff.geometry,
                                  locations = dff.index,
                                  color = dff.terr_colour,
                                  color_discrete_map=({colour : colour}),
                                  opacity = 0.7
                                  ).data[0]
        )
        print(str(            px.choropleth_mapbox(dff,
                                  geojson = dff.geometry,
                                  locations = dff.index,
                                  color = dff.terr_colour,
                                  color_discrete_map=({colour : colour}),
                                  opacity = 0.7
                                  ).data[0]
                 )
             )

    fig.update_layout(uirevision = 'Retain user zoom preferences')  

    debug_output = 'index triggered: ' + str(legendTriggered) + ' color selected: ' +str(selectedColor) + ' postcode: ' + str(postcode) + ' postcode list: ' + str(postcode_list) + ' ' + str(postcode_colour_d)           
    
    return fig, None, postcode_colour_d, html.Pre(debug_output) , clear_map_data
    

if __name__ == "__main__": app.run_server(debug=False, host='0.0.0.0', port=8050)


