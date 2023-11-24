# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import utils
import math
# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]

app = Dash(__name__, external_stylesheets=external_stylesheets)



# Use the following function when accessing the value of 'my-slider'
# in callbacks to transform the output value to logarithmic

dict_regression_model = utils.read_csv('variation_2_k_fold.csv')

def calculate_ln_poc(temp_value, pop_value, fdi_value):
    '''
    Calculates from value that IS NOT TRANSFORMED YET ! 
    '''
    ypred = dict_regression_model['CONSTANT'] + dict_regression_model['TEMP'] * transform_temp(temp_value) \
        + dict_regression_model['POP'] * transform_pop(pop_value) + dict_regression_model['FDI'] * fdi_value
    
    return ypred

def transform_pop(value):
    return  math.log(value, math.e)

def transform_temp(value):
    return  math.log(value, math.e)

def transform_fdi(value):
    return value

# ['POP','TEMP', 'FDI']
app.layout = dbc.Container([
        dbc.Row(
            children = [
                dbc.Col(
                        html.Div([ 
                            dcc.Graph(id="graph_poc_temp"),
                        ])
                    ),
                dbc.Col(
                    html.Div([ 
                            dcc.Graph(id="graph_poc_pop"),
                        ])
                    ),
                dbc.Col(html.Div([ 
                            dcc.Graph(id="graph_poc_fdi"),
                        ])
                    )
            ]
        ),
        dbc.Row(
            children = [
                dbc.Col(
                    children=[html.Div(children='Production of crop regression model'),
                            html.Div(id='pocOutput')
                            ])
            ]   
        ),
        dbc.Row(
            children= [
                html.Div(id='tempOutputSlider', style={'marginTop': 20}),
                dcc.Slider(0, 50, 0.01, id='tempInput', marks={i: f'{i}' for i in range(0,210,10)}, value=25, updatemode='drag'),                
            ]
            ),
        dbc.Row(
            children= [
                html.Div(id='popOutputSlider', style={'marginTop': 20}),
                dcc.Slider(0, 200, 0.01, id='popInput', marks={i: f'{i}' for i in range(0,210,10)}, value=30, updatemode='drag'),
                
            ]
        ),
        dbc.Row(
            children= [
                html.Div(id='fdiOutputSlider', style={'marginTop': 20}),
                dcc.Slider(-1, 2, 0.01, id='fdiInput', marks={round(i/10 - 1, 2): '{:.2f}'.format(round(i/10 - 1, 2)) for i in range(32)}, value=0.1, updatemode='drag')
                
            ]
        ),
    ]
)

@callback(
    Output('pocOutput', 'children'),
    [Input('tempInput', 'value'),
     Input('popInput', 'value'),
     Input('fdiInput', 'value')]
)
def display_poc(temp_value, pop_value, fdi_value):
    poc = calculate_ln_poc(temp_value, pop_value, fdi_value)
    return f'Predicted Production of Cereal: {poc}'

@callback(Output('tempOutputSlider', 'children'),
              Input('tempInput', 'value'))
def display_temp_slider(value):
    return 'Temperature : Linear Value: {} | \
            Log Value: {:0.2f}'.format(value, transform_temp(value))
            
@callback(Output('popOutputSlider', 'children'),
              Input('popInput', 'value'))
def display_pop_slider(value):
    return 'Population (million) : Linear Value: {} | \
            Log Value: {:0.2f}'.format(value, transform_pop(value))
            
@callback(Output('fdiOutputSlider', 'children'),
              Input('fdiInput', 'value'))
def display_fdi_slider(value):
    return 'Foreign directed investment in Iran against GDP (1 represents GDP): Linear Value: {:0.2f}'.format(value, transform_fdi(value))

@callback(
    Output('graph_poc_temp', 'figure'),
    [Input('tempInput', 'value'),
     Input('popInput', 'value'),
     Input('fdiInput', 'value')]
)
def update_graph_poc_temp(temp_value, pop_value, fdi_value):
    
    #temp value is TEMPERATURE, NOT TEMPERORARY
    min_temp = 0.1
    max_temp = 50
    temp_step_size = (max_temp - min_temp)/100
    
    new_temp_value = min_temp
    dict_graph_data = {
        'ln(POC)': [],
        'ln(TEMP)': []
    }

    for i in range(100):
        
        ln_poc_predicted = calculate_ln_poc(new_temp_value, pop_value, fdi_value)        
        
        dict_graph_data['ln(POC)'].append(ln_poc_predicted)
        dict_graph_data['ln(TEMP)'].append(new_temp_value)
        new_temp_value += temp_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="ln(TEMP)", y="ln(POC)")
    return fig

@callback(
    Output('graph_poc_pop', 'figure'),
    [Input('tempInput', 'value'),
     Input('popInput', 'value'),
     Input('fdiInput', 'value')]
)
def update_graph_poc_pop(temp_value, pop_value, fdi_value):
    
    
    min_pop = 0.1
    max_pop = 210
    pop_step_size = (max_pop - min_pop)/100
    
    new_pop_value = min_pop
    dict_graph_data = {
        'ln(POC)': [],
        'ln(POP)': []
    }

    for i in range(100):
        
        ln_poc_predicted = calculate_ln_poc(temp_value, new_pop_value, fdi_value)        
        
        dict_graph_data['ln(POC)'].append(ln_poc_predicted)
        dict_graph_data['ln(POP)'].append(new_pop_value)
        new_pop_value += pop_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="ln(POP)", y="ln(POC)")
    return fig

@callback(
    Output('graph_poc_fdi', 'figure'),
    [Input('tempInput', 'value'),
     Input('popInput', 'value'),
     Input('fdiInput', 'value')]
)
def update_graph_poc_fdi(temp_value, pop_value, fdi_value):
    
    
    min_fdi = -1
    max_fdi = 2
    fdi_step_size = (max_fdi - min_fdi)/100
    
    new_fdi_value = min_fdi
    dict_graph_data = {
        'ln(POC)': [],
        'fdi': []
    }

    for i in range(100):
        
        ln_poc_predicted = calculate_ln_poc(temp_value, pop_value, new_fdi_value)        
        
        dict_graph_data['ln(POC)'].append(ln_poc_predicted)
        dict_graph_data['fdi'].append(new_fdi_value)
        new_fdi_value += fdi_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="fdi", y="ln(POC)")
    return fig

if __name__ == '__main__':
    app.run(debug=True)
