from dash import Dash, html, dash_table, dcc

from dash.dependencies import Input, Output
import plotly.express as px

import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input

'''
This file serves as a documentation / tutorial for other people to maintain yet unfamiliar with dash library
'''
# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
app = Dash(__name__)

# # App layout
# app.layout = html.Div([
#     html.Div(children='My First App with Data and a Graph'),
#     dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    
#     #instantiate histogram from plotly express
#     # x is column, y is also column name, averages is aggergate function
#     dcc.Graph(figure=px.histogram(df, x='continent', y='lifeExp', histfunc='avg'))
# ])

# App layout this includes control and callbacks
# app.layout = html.Div([
    
#     html.Div(children='My First App with Data, Graph, and Controls'),
#     html.Hr(),
    
#     #dcc are control components - radio button is controlling the barchart
#     dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
    
    
#     dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    
#     # Raio item is controlling this
#     dcc.Graph(figure={}, id='controls-and-graph')
# ])

# # Add controls to build the interaction
# @callback(
#     #Inputs into here, it passes to the dcc.graph  (matches  the id of the dcc graph)
#     Output(component_id='controls-and-graph', component_property='figure'),
    
#     #layout actually passes value to this input (look at the component_id matches the id of the dcc. Value extracted from)
#     Input(component_id='controls-and-radio-item', component_property='value')
# )
# def update_graph(col_chosen):
#     fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
#     return fig #right now dcc.Graph(figure=fig)

# # Input connected to dcc.RadioItems. component_property='value' -> takes the VALUE out of dcc
# # Output connected to dcc.Graph  -> component_proprety='figure' -> give figure to dcc.Graph -> update_graph is called

# # How is update_graph connected? callback is a decorator for the update_graph function. update_graph looks for Output component_id and then interacts with that using the Input['component_property']


## Styling your app #stylomilo

# # Incorporate data
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# # Initialize the app - incorporate css
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = Dash(__name__, external_stylesheets=external_stylesheets)

# # App layout
# app.layout = html.Div([
#     html.Div(className='row', children='My First App with Data, Graph, and Controls',
#              style={'textAlign': 'center', 'color': 'blue', 'fontSize': 30}),

#     html.Div(className='row', children=[
#         dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'],
#                        value='lifeExp',
#                        inline=True,
#                        id='my-radio-buttons-final')
#     ]),

#     html.Div(className='row', children=[
#         html.Div(className='six columns', children=[
#             dash_table.DataTable(data=df.to_dict('records'), page_size=11, style_table={'overflowX': 'auto'})
#         ]),
#         html.Div(className='six columns', children=[
#             dcc.Graph(figure={}, id='histo-chart-final')
#         ])
#     ])
# ])

# # Add controls to build the interaction
# @callback(
#     Output(component_id='histo-chart-final', component_property='figure'),
#     Input(component_id='my-radio-buttons-final', component_property='value')
# )
# def update_graph(col_chosen):
#     fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
#     return fig



# Run the app
if __name__ == '__main__':
    app.run(debug=True)
