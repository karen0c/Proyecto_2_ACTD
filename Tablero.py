# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:53:01 2023

@author: karen
"""
#pip install dash-bootstrap-components
import dash
from dash import dcc  # dash core components
#from dash import html # dash html components
from dash.dependencies import Input, Output
from pgmpy.inference import VariableElimination
import plotly.express as px

from pgmpy.readwrite import BIFReader

import pandas as pd

columnas = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
datos_iniciales = pd.read_csv('https://raw.githubusercontent.com/karen0c/Proyecto_1_ACTD/main/processed.cleveland.data',header=None, names=columnas, na_values="?")

datos_iniciales = datos_iniciales.dropna().reset_index(drop=True) #elimina filas con valores faltantes

for i in datos_iniciales.index:
  if datos_iniciales.loc[i, "num"] == 0:
    datos_iniciales.loc[i, "num"] = 0
  else:
    datos_iniciales.loc[i, "num"] = 1

datos = datos_iniciales.copy()
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


# Read model from BIF file 
reader = BIFReader("ModeloK2.bif")
modelo = reader.get_model()

# Print model 
print(modelo)



from pgmpy.estimators import BayesianEstimator
emv = BayesianEstimator(model=modelo, data=datos)

modelo.fit(data=datos, estimator = BayesianEstimator)   
modelo.check_model()

infer = VariableElimination(modelo)



#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

import dash_bootstrap_components as dbc
import dash_html_components as html

# Crear la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define el diseño de la aplicación Dash con dos pestañas

# Definir el contenido de la primera pestaña

# Obtener la base de datos
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:proyecto2@proy2database.czhxhldkmqh7.us-east-1.rds.amazonaws.com/processed_cleveland')
datos_iniciales.to_sql('data', con=engine, if_exists='replace', index=False)

tabla=pd.read_sql('SELECT * FROM data', engine)

# Diseñar visualizaciones

#Gráfico 1

M_sin = 0
M_con = 0
F_sin = 0
F_con = 0

for i in tabla.index:
    if tabla.loc[i,'sex']==1:
        if tabla.loc[i,'num']==0:
            M_sin = M_sin+1
        else:
            M_con = M_con+1
    else:
        if tabla.loc[i,'num']==0:
            F_sin = F_sin+1
        else:
            F_con = F_con+1
        
df1 = pd.DataFrame({
    "Sexo": ["Hombre", "Hombre", "Mujer", "Mujer"],
    "Número de personas": [M_sin, M_con, F_sin, F_con],
    "Diagnóstico": ["Sin presencia de enfermedad", "Con presencia de enfermedad", "Sin presencia de enfermedad", "Con presencia de enfermedad"]
})

fig1 = px.bar(df1, x='Sexo', y='Número de personas', color='Diagnóstico', barmode='group', color_discrete_sequence=['greenyellow', 'red'])
fig1.update_layout(legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01))
fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")  


# Gráfico 2
valores2 = [0,0,0,0,0,0,0,0,0,0,0,0]
for i in tabla.index:
    if tabla.loc[i, 'chol']<=200:
        if tabla.loc[i, 'age']<40:
            valores2[0] = valores2[0]+1
        elif tabla.loc[i, 'age']>=40 and tabla.loc[i, 'age']<50:
            valores2[1] = valores2[1]+1
        elif tabla.loc[i, 'age']>=50 and tabla.loc[i, 'age']<60:
            valores2[2] = valores2[2]+1
        else:
            valores2[3] = valores2[3]+1
    elif tabla.loc[i, 'chol']>200 and tabla.loc[i, 'chol']<=240:
            if tabla.loc[i, 'age']<40:
                valores2[4] = valores2[4]+1
            elif tabla.loc[i, 'age']>=40 and tabla.loc[i, 'age']<50:
                valores2[5] = valores2[5]+1
            elif tabla.loc[i, 'age']>=50 and tabla.loc[i, 'age']<60:
                valores2[6] = valores2[6]+1
            else:
                valores2[7] = valores2[7]+1
    else:
        if tabla.loc[i, 'age']<40:
            valores2[8] = valores2[8]+1
        elif tabla.loc[i, 'age']>=40 and tabla.loc[i, 'age']<50:
            valores2[9] = valores2[9]+1
        elif tabla.loc[i, 'age']>=50 and tabla.loc[i, 'age']<60:
            valores2[10] = valores2[10]+1
        else:
            valores2[11] = valores2[11]+1
            
df2 = pd.DataFrame({
    "Edad": ["Menos de 40 años", "Entre 41 y 50 años", "Entre 51 y 60 años", "Más de 60 años", "Menos de 40 años", "Entre 41 y 50 años", "Entre 51 y 60 años", "Más de 60 años", "Menos de 40 años", "Entre 41 y 50 años", "Entre 51 y 60 años", "Más de 60 años"],
    "Número de personas": valores2,
    "Nivel": ["Nivel de colesterol deseable", "Nivel de colesterol deseable", "Nivel de colesterol deseable", "Nivel de colesterol deseable", "Nivel de colesterol límite", "Nivel de colesterol límite", "Nivel de colesterol límite", "Nivel de colesterol límite", "Nivel de colesterol alto", "Nivel de colesterol alto", "Nivel de colesterol alto", "Nivel de colesterol alto"]
})

fig2 = px.bar(df2, x='Edad', y='Número de personas', color='Nivel', barmode='group', color_discrete_sequence=['greenyellow', 'orange', 'red'])
fig2.update_layout(legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01))
fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")  


# Gráfico 3
valores3 = [0,0,0,0,0,0,0,0]
for i in tabla.index:
    if tabla.loc[i, 'num'] == 0:
        if tabla.loc[i, 'age']<40:
            valores3[0] = valores3[0]+1
        elif tabla.loc[i, 'age']>=40 and tabla.loc[i, 'age']<50:
            valores3[1] = valores3[1]+1
        elif tabla.loc[i, 'age']>=50 and tabla.loc[i, 'age']<60:
            valores3[2] = valores3[2]+1
        else:
            valores3[3] = valores3[3]+1
    else:
        if tabla.loc[i, 'age']<40:
            valores3[4] = valores3[4]+1
        elif tabla.loc[i, 'age']>=40 and tabla.loc[i, 'age']<50:
            valores3[5] = valores3[5]+1
        elif tabla.loc[i, 'age']>=50 and tabla.loc[i, 'age']<60:
            valores3[6] = valores3[6]+1
        else:
            valores3[7] = valores3[7]+1

porcentajes = [round(valores3[0]/(valores3[0]+valores3[4]),3), round(valores3[1]/(valores3[1]+valores3[5]),3), round(valores3[2]/(valores3[2]+valores3[6]),3), round(valores3[3]/(valores3[3]+valores3[7]),3), round(valores3[4]/(valores3[0]+valores3[4]),3), round(valores3[5]/(valores3[1]+valores3[5]),3), round(valores3[6]/(valores3[2]+valores3[6]),3), round(valores3[7]/(valores3[3]+valores3[7]),3)]   


df3 = pd.DataFrame({
    "Edad": ["Menos de 40 años", "Entre 41 y 50 años", "Entre 51 y 60 años", "Más de 60 años", "Menos de 40 años", "Entre 41 y 50 años", "Entre 51 y 60 años", "Más de 60 años"],
    "Proporción de personas": porcentajes,
    "Diagnóstico": ["Sin presencia de enfermedad", "Sin presencia de enfermedad", "Sin presencia de enfermedad", "Sin presencia de enfermedad","Con presencia de enfermedad", "Con presencia de enfermedad", "Con presencia de enfermedad", "Con presencia de enfermedad"]
})

fig3 = px.bar(df3, x='Edad', y='Proporción de personas', color='Diagnóstico', barmode='stack', color_discrete_sequence=['greenyellow', 'red'])
fig3.update_layout(legend=dict(yanchor="bottom", y=1, xanchor="left", x=0.01))
fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")  

#Crear gráficos
tab1_content = dbc.Card(
    dbc.CardBody([
    html.H2('¿Qué dicen los datos?'),
    html.P(children='Queremos proporcionarte información relevante sobre la enfermedad cardiaca, basada en estadísticas del conjunto de datos Heart Disease disponible en el repositorio de la Universidad de California en Irvine. Esperamos que esta información sea útil para ti.' ),
    html.Div([
  
    html.Div([
        dcc.Graph(id='grafico1', figure = fig1)],
        style={'width': '33%', 'display': 'inline-block'}),
        
    html.Div([
        dcc.Graph(id='grafico2', figure = fig3)],
        style={'width': '33%', 'display': 'inline-block'}),
        
    html.Div([
        dcc.Graph(id='grafico3', figure = fig2)],
        style={'width': '33%', 'display': 'inline-block'})
    ], style={'display': 'flex'})
    
],  style={'margin': '30px'}))


tab2_content=dbc.Card(
    dbc.CardBody([
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
        )],  style={'width': '50%', 'marginRight': '50px'}),   
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
            )],  style={'width': '50%','marginRight': '50px'}),
            
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
            )],  style={'width': '50%', 'marginRight': '50px'}),
            
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
    html.H6("A continuación te presentamos la probabilidad de tener o no una enfermedad cardiaca:"),
    html.Br(),
    html.Div([
        html.Div(
        dcc.Graph(id='grafico'),
         style={'width': '60%'}),
        html.Div([
            html.Br(),
            html.Div(id='recomendación')],
            style={'textAlign': 'left','marginTop':'150px','width': '30%'})], style={'display': 'flex','width': '100%'}, className='row'),
    ],  style={'margin': '30px'}
))

# Definir las pestañas
tabs = dbc.Tabs([
    dbc.Tab(label='Datos de interés', tab_id='tab-1', children=[tab1_content]),
    dbc.Tab(label='Obten aquí tu probabilidad', tab_id='tab-2', children=[tab2_content]),
], id='tabs', active_tab='tab-1', className="nav nav-tabs")


# Definir el diseño de la aplicación
app.layout = dbc.CardBody([
       html.Div([
       html.Img(src='https://archive-beta.ics.uci.edu/static/public/45/Thumbnails/Large.jpg?30',
                 style={'height': '70%', 'width': 'auto','max-width': '100%'}),
        html.H1(children='¿Acaso tengo una enfermedad cardiaca?',
                style={'textAlign': 'center', 'marginTop': '25px', "width": "80%", 'fontSize': '3vw'}),
        html.Img(src='https://educacion.uniandes.edu.co/sites/default/files/Uniandes.png',
                 style={'height': '50%', 'width': 'auto', 'max-width': '100%'})
    ], style={'display': 'flex', 'height': '120px', "width": "100%"}),
    html.Br(),
    tabs
    
 ], style={'margin': '30px'})


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
     Input(component_id='input_fbs', component_property='value')
     ])
    


def update_pie_chart(input_age, input_sex, input_chol, input_trestbps, input_thal, input_fbs):
    
    valores = ['age','sex','chol','trestbps','thal','fbs']
    respuesta = [input_age, input_sex, input_chol, input_trestbps, input_thal, input_fbs]
      
    aux={}
    for i in range(0, 6):
        if respuesta[i] != '-1': 
            aux[valores[i]]= respuesta[i]
        
    if len(aux)==0:
        posterior_p = infer.query(["num"], evidence={'age':2,'sex':1,'chol':2,'trestbps':1,'thal':3,'fbs':1})

    else:
        posterior_p = infer.query(["num"], evidence=aux)
    
    num_states = modelo.get_cpds("num").state_names["num"]
    
    # valores para el gráfico de torta
    labels = ['Ausencia de enfermedad cardiaca', 'Presencia de enfermedad cardiaca']
    values = [posterior_p.values[num_states.index(0)], posterior_p.values[num_states.index(1)]]
    
    
    # Crear el objeto Pie de Plotly
    #if isinstance(values[0], (int, float))
        
    import plotly.graph_objs as go
    figura = go.Figure(go.Pie(labels=labels, 
                          values=values,
                          textfont={'size': 20},
                          #hole=0.5,
                          hoverinfo='label+percent',
                          marker=dict(colors=["greenyellow", "red"],
                                      line=dict(color='#000000', width=1)),
                          #textposition='outside',
                          pull = [0, 0.2]
                         ), layout=(go.Layout( margin=dict(l=100, r=10, t=10, b=10))                     
                        )
)
    figura.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")  
    if values[0] > 0.8 :
            recomendación = 'Como la probabilidad de que no tengas una enfermedad cardiaca es alta ('+'{:.0%}'.format(values[0]) +'), te sugerimos continuar con tus chequeos de control, teniendo en cuenta que no es urgente que consultes un médico especialista.'
    elif values[0] > 0.5:
            recomendación = 'A pesar de que la probabilidad de que tengas una enfermedad cardiaca no es tan alta ('+ '{:.0%}'.format(values[1]) +'), te sugerimos consultar a un médico especialista y así decartar que tengas una enfermedad cardiaca.'
    elif values[0] > 0.25:
            recomendación = 'De acuerdo con tus características, es probable que tengas una enfermedad cardiaca ('+'{:.0%}'.format(values[1]) +'), te sugerimos consultar a un médico especialista en el menor tiempo posible.'
    elif values[0]<=0.25: 
            recomendación = 'De acuerdo con tus características, la probabilidad de tener una enfermedad cardiaca es muy alta (' +'{:.0%}'.format(values[1]) + '), deberias consultar a un médico especialista de inmediato para confirmar esto y si es así, iniciar un tratamiento.'
    else: 
        recomendación = "Lamentamos informarle que no tenemos evidencia para el caso presentado, por lo que no podemos estimarlo."
       
    return figura, recomendación

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
