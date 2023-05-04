# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:53:01 2023

@author: karen
"""
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
from pgmpy.inference import VariableElimination


import plotly.express as px

import pandas as pd

columnas = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
datos_iniciales = pd.read_csv('https://raw.githubusercontent.com/karen0c/Proyecto_1_ACTD/main/processed.cleveland.data',header=None, names=columnas, na_values="?")

datos_iniciales = datos_iniciales.dropna().reset_index(drop=True) #elimina filas con valores faltantes
#print(datos_iniciales.head())
#print(datos_iniciales.tail())

# crear unos nuevos datos donde guardaremos la información con los datos discretizados
datos = datos_iniciales
for i in range(0,297):
  if datos_iniciales.loc[i, 'age'] <= 40:
    datos.loc[i, 'age'] = 1
  elif datos_iniciales.loc[i, 'age'] > 40 and datos_iniciales.loc[i, 'age'] <= 50:
    datos.loc[i, 'age'] = 2
  elif datos_iniciales.loc[i, 'age'] > 50 and datos_iniciales.loc[i, 'age'] <= 60:
    datos.loc[i, 'age'] = 3
  else:
    datos.loc[i, 'age'] = 4

for i in range(0,297):
  if datos_iniciales.loc[i, "trestbps"] <= 120:
    datos.loc[i, "trestbps"] = 1
  elif datos_iniciales.loc[i, "trestbps"] > 120 and datos_iniciales.loc[i, "trestbps"] <= 139:
    datos.loc[i, "trestbps"] = 2
  elif datos_iniciales.loc[i, "trestbps"] >= 140 and datos_iniciales.loc[i, "trestbps"] <= 159:
    datos.loc[i, "trestbps"] = 3
  elif datos_iniciales.loc[i, "trestbps"] >= 160 and datos_iniciales.loc[i, "trestbps"] <= 179:
    datos.loc[i, "trestbps"] = 4
  else:
    datos.loc[i, "trestbps"] = 5
    
for i in range(0,297):
  if datos_iniciales.loc[i, "chol"] <= 200:
    datos.loc[i, "chol"] = 1
  elif datos_iniciales.loc[i, "chol"] > 200 and datos_iniciales.loc[i, "chol"] < 240:
    datos.loc[i, "chol"] = 2
  else:
    datos.loc[i, "chol"] = 3
    
for i in range(0,297):
  if datos_iniciales.loc[i, "thalach"] <= 120:
    datos.loc[i, "thalach"] = 1
  elif datos_iniciales.loc[i, "thalach"] > 120 and datos_iniciales.loc[i, "thalach"] <= 140:
    datos.loc[i, "thalach"] = 2
  elif datos_iniciales.loc[i, "thalach"] > 140 and datos_iniciales.loc[i, "thalach"] < 160:
    datos.loc[i, "thalach"] = 3
  else:
    datos.loc[i, "thalach"] = 4
    
for i in range(0,297):
  if datos_iniciales.loc[i, "oldpeak"] <= 1:
    datos.loc[i, "oldpeak"] = 1
  elif datos_iniciales.loc[i, "oldpeak"] > 1 and datos_iniciales.loc[i, "oldpeak"] <= 2:
    datos.loc[i, "oldpeak"] = 2
  else:
    datos.loc[i, "oldpeak"] = 3
    
for i in datos_iniciales.index:
  if datos_iniciales.loc[i, "num"] == 0:
    datos.loc[i, "num"] = 0
  else:
    datos.loc[i, "num"] = 1



#pip install pgmpy 

#pip install numpy

from pgmpy.models import BayesianNetwork
#from pgmpy.factors.discrete import TabularCPD

modelo = BayesianNetwork([("sex", "chol"), ("age", "chol"), ("age", "fbs"),("thal", "trestbps"), ("chol", "num"),("fbs", "trestbps"), ("trestbps", "num"),("num", "ca"),("num", "thalach"),("num", "exang"),("num", "restecg"),("exang", "cp"),("cp", "oldpeak"),( "restecg","oldpeak"),("restecg","slope")])

from pgmpy.estimators import BayesianEstimator

emv = BayesianEstimator(model=modelo, data=datos)

modelo.fit(data=datos, estimator = BayesianEstimator) 

#for i in modelo.nodes(): 
 # print(modelo.get_cpds(i)) 

infer = VariableElimination(modelo)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Define el diseño de la aplicación Dash con dos pestañas
app.layout1 = html.Div(['una parte'
   
    ])

app.layout2 = html.Div(
    [ 
       html.Div([
       html.Img(src='https://archive-beta.ics.uci.edu/static/public/45/Thumbnails/Large.jpg?30',
                 style={'height': '70%', 'width': 'auto','max-width': '100%'}),
        html.H1(children='¿Acaso tengo una enfermedad cardiaca?',
                style={'textAlign': 'center', 'marginTop': '25px', "width": "80%", 'fontSize': '3vw'}),
        html.Img(src='https://educacion.uniandes.edu.co/sites/default/files/Uniandes.png',
                 style={'height': '50%', 'width': 'auto', 'max-width': '100%'})
    ], style={'display': 'flex', 'height': '120px', "width": "100%"}),
    html.Br(),
    html.P(children='Aquí te brindamos una herramienta que puedes utilizar desde casa donde te mostramos qué tan probable es que tengas una enfermedad cardiaca y con ello, queremos ayudarte a tomar la decisión de consultar o no a tu médico de acuerdo con tus síntomas.' ),
    html.Br(),
    
    html.H6("Ingresa el valor de la información que tengas disponible:"),
    
    html.Div([
        html.Div(['Edad:',
                dcc.Dropdown(
                    id='input_age',
                    options=[
                        {'label': 'Entre 25 y 40 años', 'value': 1},
                        {'label': 'Entre 41 y 50 años', 'value': 2},
                        {'label': 'Entre 51 y 60 años', 'value': 3},
                        {'label': 'Mayor a 60 años', 'value': 4}
            ],
            value='-1'
        )],  style={'width': '50%'}),   
        html.Div(['Sexo:',
        dcc.Dropdown(
            id='input_sex',
            options=[
                {'label': 'Mujer', 'value': 0},
                {'label': 'Hombre', 'value': 1}
            ],
            value='-1'
        )], style={'width': '50%'})],  style={'display': 'flex', "width": "100%"}),
    
    html.Div([
            html.Div(['Nivel de colesterol:',
            dcc.Dropdown(
                id='input_chol',
                options=[
                    {'label': 'Menor o igual a 200', 'value': 1},
                    {'label': 'Entre 201 y 240', 'value': 2},
                    {'label': 'Mayor o igual a 240', 'value': 3},
                ],
                value='-1'
            )],  style={'width': '50%'}),
            
            html.Div(['Nivel de presión arterial en reposo:',
            dcc.Dropdown(
                id='input_trestbps',
                options=[
                    {'label': 'Menor o igual a 120', 'value': 1},
                    {'label': 'Entre 121 y 139', 'value': 2},
                    {'label': 'Entre 140 y 159', 'value': 3},
                    {'label': 'Entre 160 y 179', 'value': 4},
                    {'label': 'Mayor o igual a 159', 'value': 5},
                ],
                value='-1'
            )], style={'width': '50%'})],  style={'display': 'flex', "width": "100%"}),
    
    html.Div([
            html.Div(['En caso de presentar talasemia, indica el tipo:',
            dcc.Dropdown(
                id='input_thal',
                options=[
                    {'label': 'Normal', 'value': 3},
                    {'label': 'Defecto fijo', 'value': 6},
                    {'label': 'Defecto reversible', 'value': 7},
                ],
                value='-1'
            )],  style={'width': '50%'}),
            
            html.Div(['El nivel de azucar en la sangre en ayunas es mayor a 120 mg/dl:',
            dcc.Dropdown(
                id='input_fbs',
                options=[
                    {'label': 'Sí', 'value': 1},
                    {'label': 'No', 'value': 0},
                   
                ],
                value='-1'
            )], style={'width': '50%'})],  style={'display': 'flex', "width": "100%"}),
    html.Br(),
    html.H6("A continuación te presentamos la probabilidad de tener cierto tipo de enfermedad cardiaca:"),
    html.Br(),
    html.Div([
        html.Div(
        dcc.Graph(id='grafico'),
         style={'width': '50%'}),
        html.Div([
            html.Br(),
            html.Div(id='recomendación')],
            style={'textAlign': 'center','marginTop':'150px','width': '50%'})], style={'display': 'flex','width': '100%'}, className='row'),
    ],  style={'margin': '30px'}
)


@app.callback(
    [Output(component_id='grafico', component_property='figure'),
     Output(component_id='recomendación', component_property='children')]
     ,
    [
     Input(component_id='input_age', component_property='value'),
     Input(component_id='input_sex', component_property='value'),
     Input(component_id='input_chol', component_property='value'),
     Input(component_id='input_trestbps', component_property='value'),
     Input(component_id='input_thal', component_property='value'),
     Input(component_id='input_fbs', component_property='value'),
     ]
    
)


def update_pie_chart(input_age, input_sex, input_chol, input_trestbps, input_thal, input_fbs):
    
    valores = ['age','sex','chol','trestbps','thal','fbs']
    respuesta = [input_age, input_sex, input_chol, input_trestbps, input_thal, input_fbs]
      
    aux={}
    for i in range(0, 6):
        if respuesta[i] != '-1': 
            aux[valores[i]]= respuesta[i]
        
    if len(aux)==0:
        posterior_p = infer.query(["num"], evidence={'age':1,'sex':1,'chol':1,'trestbps':1,'thal':3,'fbs':1})
    else:
        posterior_p = infer.query(["num"], evidence=aux)
    
    num_states = modelo.get_cpds("num").state_names["num"]
    
    # Ejemplo de valores para el gráfico de torta
    labels = ['Sin presencia de enfermedad', 'Presencia de enfermedad cardiaca']
    values = [posterior_p.values[num_states.index(0)], posterior_p.values[num_states.index(1)]]
    
    # Crear el objeto Pie de Plotly
    #if isinstance(values[0], (int, float)):
    figura = px.pie(labels=labels, 
                        values=values,
                        hover_name=labels,
                        color_discrete_sequence=["green", "red"]
                        )
    if values[0] > 0.8 :
            recomendación = 'Como la probabilidad de que no tengas una enfermedad cardiaca es alta, te sugerimos continuar con tus chequeos de control, teniendo en cuenta que no es urgente que consultes un médico especialista.'
    elif values[0] > 0.5:
            recomendación = 'A pesar de que la probabilidad de que tengas una enfermedad cardiaca no es tan alta, te sugerimos consultar a un médico especialista y así decartar que tengas una enfermedad cardiaca.'
    elif values[0] > 0.25:
            recomendación = 'De acuerdo con tus características, es probable que tengas una enfermedad cardiaca, te sugerimos consultar a un médico especialista en el menor tiempo posible.'
    elif values[0]<=0.25: 
            recomendación = 'De acuerdo con tus características, la probabilidad de tener una enfermedad cardiaca es muy alta, deberias consultar a un médico especialista de inmediato para confirmar esto y si es así, iniciar un tratamiento.'
    else: 
        recomendación = "Lamentamos informarle que no tenemos evidencia para el caso presentado, por lo que no podemos estimarlo."
       
    return figura, recomendación




if __name__ == '__main__':
    app.run_server(debug=True)
    
    
