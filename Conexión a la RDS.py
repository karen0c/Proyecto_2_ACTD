# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:32:54 2023

@author: karen
"""
import pandas as pd

columnas = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num"]
datos_iniciales = pd.read_csv('https://raw.githubusercontent.com/karen0c/Proyecto_1_ACTD/main/processed.cleveland.data',header=None, names=columnas, na_values="?")
datos_iniciales = datos_iniciales.dropna().reset_index(drop=True) #elimina filas con valores faltantes


for i in datos_iniciales.index:
  if datos_iniciales.loc[i, "num"] == 0:
    datos_iniciales.loc[i, "num"] = 0
  else:
    datos_iniciales.loc[i, "num"] = 1

from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:proyecto2@proy2database.czhxhldkmqh7.us-east-1.rds.amazonaws.com/processed_cleveland')
datos_iniciales.to_sql('data', con=engine, if_exists='replace', index=False)

pd.read_sql('SELECT * FROM data', engine)
