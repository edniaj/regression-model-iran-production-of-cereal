from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import utils
import math


'''
    This file is the main file for the prediction model dashboard
'''

external_stylesheets = [dbc.themes.CERULEAN]

app = Dash(__name__, external_stylesheets=external_stylesheets)

dict_regression_model = utils.read_csv('variation_2_k_fold.csv')

''' 
Utility functions

calculate_ln_poc(temp_value: float, pop_value:float, fdi_value:float) -> float:
    This is use to calculate production of cereal based on the best multi-linear regression model we have.

transform_pop(value:float) -> float
    We will use log(pop) instead of pop for our muilti-linear regression model
    
 
transform_temp(value:float) -> float:
    We will use log(temp) instead of pop for our muilti-linear regression model

transform_fdi(value:float) -> float:
    We don't have to transform anything but will leave this function for modularity sake

untransform_poc(ln_poc:float) -> float:
    After we get the predicted POC value which will be Ln(POC), we will reverse it so that we can plot POC against variables
'''
def calculate_ln_poc(temp_value: float, pop_value:float, fdi_value:float) -> float:
    '''
    Calculates from value that IS NOT TRANSFORMED YET ! 
    '''
    ypred = dict_regression_model['CONSTANT'] + dict_regression_model['TEMP'] * transform_temp(temp_value) \
        + dict_regression_model['POP'] * transform_pop(pop_value) + dict_regression_model['FDI'] * fdi_value
    
    return ypred

def transform_pop(value:float) -> float:
    return  math.log(value, math.e)

def transform_temp(value:float) -> float:
    return  math.log(value, math.e)

def transform_fdi(value:float) -> float:
    return value

def untransform_poc(ln_poc:float) -> float:
    reverse_transform_poc = ln_poc * math.e
    poc = reverse_transform_poc * 100000
    return poc

'''
Brief layout of the app
<Container>
    <H2> title </H2>
    
    <Card>
        <CardHeader>
        
        </CardHeader>
        
        <CardBody>
            <Row>
                <InputSlider/>
            </Row>
            <Row>
                <InputSlider/>
            </Row>
            <Row>
                <InputSlider/>
            </Row>
        </CardBody>
    <Card/>
    
    Mobile responsive Div
    <Div> 
        <Graph>
        <Graph>
        <Graph>
        <Graph>
        <Graph>
        <Graph>
    </Div>

<Container>
'''

app.layout = dbc.Container([
        dbc.Row(
            children=
            [html.H2(children='Prediction model for Production of Cereal(POC) in Iran', style={'textAlign': 'center', 'color': 'blue','margin':30 }),
            html.Div( children='We have built a multi-linear regression model for the production of cereal (POC) against other variables \
                such as temperature (TEMP), population (POP) and foreign directed investment against GDP(FDI)', style={'textAlign': 'left', 'color':  'black', 'fontSize': 15, 'margin':20})]
            )
        ,
        dbc.Card(
            children=[
                dbc.CardHeader(html.H4("Calculator", className="text-center", )),

                dbc.CardBody(
                    children=[
                dbc.Row(
                    children = [html.Div(id='pocOutput', style={'textAlign':'center', 'fontSize':30}),
                                html.Div(id='poc_million_output', style={'textAlign':'center', 'fontSize':30})]
                ),                
                dbc.Row(
                    children= [
                        html.Div(id='tempOutputSlider', style={'marginTop': 20}),
                        dcc.Slider(0, 50, 0.01, id='temp_input', marks={i: f'{i}' for i in range(0,210,10)}, value=25, updatemode='drag'),                
                    ]
                ),
                dbc.Row(
                    children= [
                        html.Div(id='popOutputSlider', style={'marginTop': 20}),
                        dcc.Slider(0, 200, 0.01, id='pop_input', marks={i: f'{i}' for i in range(0,210,10)}, value=30, updatemode='drag'),        
                    ]
                ),
                dbc.Row(
                    children= [
                        html.Div(id='fdiOutputSlider', style={'marginTop': 20}),
                        dcc.Slider(-1, 2, 0.01, id='fdi_input', marks={round(i/10 - 1, 2): '{:.2f}'.format(round(i/10 - 1, 2)) for i in range(32)}, value=0.1, updatemode='drag')
                    ]
                ),
            ]),]
        ),
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
                        html.Div([ 
                            dcc.Graph(id="graph_poc_temp_untransform"),
                        ])
                    ),
                dbc.Col(
                    html.Div([ 
                            dcc.Graph(id="graph_poc_pop_untransform"),
                        ])
                    ),
                dbc.Col(html.Div([ 
                            dcc.Graph(id="graph_poc_fdi_untransform"),
                        ])
                    )
            ]
        ),    
])


################### Scripts to connect between the input sliders and the graphs 
'''
Decorators
@callback(List[Input(id, value), Output(id,value)]) 
    These are decorators to connect the input and the function (based on component_id), and to connect the output to the function (based on component_id)

Functions
display_million_poc(temp_value:float, pop_value:float, fdi_value:float) -> str
    Display the production of Cereal inside the calculator

display_poc(temp_value:float, pop_value:float, fdi_value:float) -> str
    Display the production of Cereal ln(POC) generated by linear regression inside the calculator

display_temp_slider(value:float) -> str
    Display the value of the temperature input value to be right above the slider
    
display_pop_slider(value:float) -> str
    Display the value of the population input value to be right above the slider
    
display_fdi_slider(value:float) -> str
    Display the of the fdi input value to be right above the slider    
    
update_graph_poc_temp(temp_value:float, pop_value:float, fdi_value:float) -> px.line
    Update the graph of ln(poc) vs ln(temp)
update_graph_poc_pop(temp_value:float, pop_value:float, fdi_value:float) -> px.line
    Update the graph of ln(poc) vs ln(pop)
    
update_graph_poc_pop(temp_value:float, pop_value:float, fdi_value:float) -> px.line
    Update the graph of ln(POC) vs ln(pop)
    
update_graph_poc_temp_untransform(temp_value:float, pop_value:float, fdi_value:float ) -> px.line:
    Update the graph of POC vs TEMP

update_graph_poc_fdi_untransform(temp_value:float, pop_value:float, fdi_value:float ) -> px.line
    Update the graph of POC vs FDI

update_graph_poc_pop_untransform(temp_value:float, pop_value:float, fdi_value:float ) -> px.line
    Update the graph of POC vs POP
'''

########## Update the div html text of POC value

@callback(
    Output('poc_million_output', 'children'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def display_million_poc(temp_value:float, pop_value:float, fdi_value:float) -> str:
    ln_poc = calculate_ln_poc(temp_value, pop_value, fdi_value)
    poc = untransform_poc(ln_poc)
    # 'Foreign directed investment(1 represents GDP) :  {:0.2f}'.format(value
    return 'Predicted POC (1 ton): {:0.3f}'.format(poc)

@callback(
    Output('pocOutput', 'children'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def display_poc(temp_value:float, pop_value:float, fdi_value:float):
    poc = calculate_ln_poc(temp_value, pop_value, fdi_value)
    # 'Foreign directed investment(1 represents GDP) :  {:0.2f}'.format(value
    return 'Predicted ln(POC): {:0.3f}'.format(poc)

######## Update Input sliders value

@callback(Output('tempOutputSlider', 'children'),
              Input('temp_input', 'value'))
def display_temp_slider(value:float) -> str:
    return 'Temperature (Celsius) : {}  \
            '.format(value)
            
@callback(Output('popOutputSlider', 'children'),
              Input('pop_input', 'value'))
def display_pop_slider(value):
    return 'Population (million) :  {}  \
            '.format(value)
            
@callback(Output('fdiOutputSlider', 'children'),
              Input('fdi_input', 'value'))
def display_fdi_slider(value):
    return 'Foreign directed investment (1 represents GDP) :  {:0.2f}'.format(value, transform_fdi(value))

@callback(
    Output('graph_poc_temp', 'figure'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
######## Update graph of ln(Poc) against other regression model variables

def update_graph_poc_temp(temp_value:float, pop_value:float, fdi_value:float):
    
    
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
        dict_graph_data['ln(TEMP)'].append(math.log(new_temp_value, math.e))
        new_temp_value += temp_step_size
    
    df = pd.DataFrame(dict_graph_data) 
    fig = px.line(df, 
        x="ln(TEMP)", y="ln(POC)")
    return fig

@callback(
    Output('graph_poc_pop', 'figure'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def update_graph_poc_pop(temp_value:float, pop_value:float, fdi_value:float):
    
    
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
        dict_graph_data['ln(POP)'].append(math.log(new_pop_value, math.e))
        new_pop_value += pop_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="ln(POP)", y="ln(POC)")
    return fig

@callback(
    Output('graph_poc_fdi', 'figure'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def update_graph_poc_fdi(temp_value:float, pop_value:float, fdi_value:float) :
    
    
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

########## Upgrade graphs for untransformed variables

@callback(
    Output('graph_poc_temp_untransform', 'figure'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def update_graph_poc_temp_untransform(temp_value:float, pop_value:float, fdi_value:float ):
    
    #temp value is TEMPERATURE, NOT TEMPERORARY
    min_temp = 0.1
    max_temp = 50
    temp_step_size = (max_temp - min_temp)/100
    
    new_temp_value = min_temp
    dict_graph_data = {
        'POC': [],
        'TEMP': []
    }

    for i in range(100):
        
        ln_poc_predicted = calculate_ln_poc(new_temp_value, pop_value, fdi_value)        
        poc_predicted = untransform_poc(ln_poc_predicted)
        dict_graph_data['POC'].append(poc_predicted)
        dict_graph_data['TEMP'].append(new_temp_value)
        new_temp_value += temp_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="TEMP", y="POC")
    return fig

@callback(
    Output('graph_poc_pop_untransform', 'figure'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def update_graph_poc_pop_untransform(temp_value:float, pop_value:float, fdi_value:float ) :
    
    
    min_pop = 0.1
    max_pop = 210
    pop_step_size = (max_pop - min_pop)/100
    
    new_pop_value = min_pop
    dict_graph_data = {
        'POC': [],
        'POP': []
    }

    for i in range(100):
        ln_poc_predicted = calculate_ln_poc(temp_value, new_pop_value, fdi_value)        
        poc_predicted = untransform_poc(ln_poc_predicted)
        
        dict_graph_data['POC'].append(poc_predicted)
        dict_graph_data['POP'].append(new_pop_value )
        new_pop_value += pop_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="POP", y="POC")
    return fig

@callback(
    Output('graph_poc_fdi_untransform', 'figure'),
    [Input('temp_input', 'value'),
     Input('pop_input', 'value'),
     Input('fdi_input', 'value')]
)
def update_graph_poc_fdi_untransform(temp_value:float, pop_value:float, fdi_value:float ) :
    
    
    min_fdi = -1
    max_fdi = 2
    fdi_step_size = (max_fdi - min_fdi)/100
    
    new_fdi_value = min_fdi
    dict_graph_data = {
        'POC': [],
        'fdi': []
    }

    for i in range(100):
        
        ln_poc_predicted = calculate_ln_poc(temp_value, pop_value, new_fdi_value)        
        poc_predicted = untransform_poc(ln_poc_predicted)
        dict_graph_data['POC'].append(poc_predicted)
        dict_graph_data['fdi'].append(new_fdi_value)
        new_fdi_value += fdi_step_size
    
    df = pd.DataFrame(dict_graph_data) # replace with your own data source
    fig = px.line(df, 
        x="fdi", y="POC")
    return fig

#

if __name__ == '__main__':
    app.run(debug=True)
