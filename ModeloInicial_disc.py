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
    
for i in range(0,297):
    if datos_iniciales.loc[i, "num"] == 0:
        datos.loc[i, "num"] = 0
    else:
        datos.loc[i, "num"] = 1

# separar datos para entrenamiento y test
sep = int(0.8*len(datos))
datos_train = datos[:sep]

from pgmpy.models import BayesianNetwork
modelo = BayesianNetwork([("sex", "chol"), ("age", "chol"), ("age", "fbs"),("thal", "trestbps"), ("chol", "num"),("fbs", "trestbps"), ("trestbps", "num"),("num", "ca"),("num", "thalach"),("num", "exang"),("num", "restecg"),("exang", "cp"),("cp", "oldpeak"),( "restecg","oldpeak"),("restecg","slope")])

from pgmpy.estimators import MaximumLikelihoodEstimator 
emv = MaximumLikelihoodEstimator(model=modelo, data=datos_train)

modelo.fit(data=datos_train, estimator = MaximumLikelihoodEstimator) 
modelo.check_model()
infer = VariableElimination(modelo)

# write model to a BIF file 
from pgmpy.readwrite import BIFWriter
writer = BIFWriter(modelo)
writer.write_bif(filename='ModeloInicial.bif') 










