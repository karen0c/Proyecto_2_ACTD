from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import BIFReader
import pandas as pd

# TEST DEL MODELO INICIAL 

# Read model from BIF file
reader = BIFReader('C:/Users/kathe/OneDrive - Universidad de los Andes/Séptimo semestre/Analítica computacional/Proyecto 2/ModeloInicial.bif')
modelo1 = reader.get_model()

# Print model 
print(modelo1)

# Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
modelo1.check_model()

# Infering the posterior probability
from pgmpy.inference import VariableElimination
infer1 = VariableElimination(modelo1)

# Datos de prueba
datos = pd.DataFrame(pd.read_csv('C:/Users/kathe/OneDrive - Universidad de los Andes/Séptimo semestre/Analítica computacional/Proyecto 2/datos_discretizados.csv'))

datos_train = datos[:int(0.8*len(datos))]
datos_test = datos[int(0.8*len(datos)):]

bin_test = []
for i in datos_test.index:
    bin_test.append(datos_test.loc[i, "num"])  
    
#Obtener predicciones
num_states = modelo1.get_cpds("num").state_names["num"]
bin_modelo = []

datos_test.tail()

for i in range(237,297):
    posterior_p = infer1.query(["num"], evidence={'age': str(datos_test.loc[i, 'age']), 'sex': str(datos_test.loc[i, 'sex']),'cp': str(datos_test.loc[i, 'cp']),'trestbps': str(datos_test.loc[i, 'trestbps']),'chol': str(datos_test.loc[i, 'chol']),'fbs': str(datos_test.loc[i, 'fbs']),'restecg': str(datos_test.loc[i, 'restecg']),'thalach': str(datos_test.loc[i, 'thalach']),'exang': str(datos_test.loc[i, 'exang']),'oldpeak': str(datos_test.loc[i, 'oldpeak']),'slope': str(datos_test.loc[i, 'slope']),'ca': str(datos_test.loc[i, 'ca']),'thal': str(datos_test.loc[i, 'thal'])})
    posicion = -1
    probabilidad = -1
    
    if posterior_p.values[num_states.index('0')]>posterior_p.values[num_states.index('1')]:
        bin_modelo.append(0)
    else:
        bin_modelo.append(1)

# Obtener estadísticas
verd_positivo1 = 0
verd_negativo1 = 0
falso_positivo1 = 0
falso_negativo1 = 0

#contar el número de verdaderos positivos y negativos            
for i in range(0,60):
    if bin_test[i]==bin_modelo[i]:
        if bin_modelo[i]==0:
            verd_positivo1 = verd_positivo1+1
        else:
            verd_negativo1 = verd_negativo1+1
    else:
        if bin_modelo[i]==0:
            falso_positivo1 = falso_positivo1+1
        else:
            falso_negativo1 = falso_negativo1+1

print(verd_positivo1, verd_negativo1, falso_positivo1, falso_negativo1)

#K2Score y BicScore
from pgmpy.estimators import K2Score
k2_score1 = K2Score(data=datos_train).score(modelo1)
print(k2_score1)

from pgmpy.estimators import BicScore
bic_score1 = BicScore(data=datos_train).score(modelo1)
print(bic_score1)



# MODELO DE OTRO GRUPO

reader2 = BIFReader('C:/Users/kathe/OneDrive - Universidad de los Andes/Séptimo semestre/Analítica computacional/Proyecto 2/Modelo_grupoext.bif')
modelo2 = reader2.get_model()

print(modelo2.edges())

# Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
modelo2.check_model()

# Infering the posterior probability
infer2 = VariableElimination(modelo2)
  
#Obtener predicciones
num_states = modelo2.get_cpds("num").state_names["num"]
bin_modelo2 = []

for i in range(237,297):
    posterior_p = infer2.query(["diagnosis"], evidence={'age': str(datos_test.loc[i, 'age']), 'sex': str(datos_test.loc[i, 'sex']),'cp': str(datos_test.loc[i, 'cp']),'trestbps': str(datos_test.loc[i, 'trestbps']),'chol': str(datos_test.loc[i, 'chol']),'fbs': str(datos_test.loc[i, 'fbs']),'restecg': str(datos_test.loc[i, 'restecg']),'thalach': str(datos_test.loc[i, 'thalach']),'exang': str(datos_test.loc[i, 'exang']),'oldpeak': str(datos_test.loc[i, 'oldpeak']),'slope': str(datos_test.loc[i, 'slope']),'ca': str(datos_test.loc[i, 'ca']),'thal': str(datos_test.loc[i, 'thal'])})
    posicion = -1
    probabilidad = -1
    
    if posterior_p.values[num_states.index('0')]>posterior_p.values[num_states.index('1')]:
        bin_modelo2.append(0)
    else:
        bin_modelo2.append(1)

# Obtener estadísticas
verd_positivo2 = 0
verd_negativo2 = 0
falso_positivo2 = 0
falso_negativo2 = 0

#contar el número de verdaderos positivos y negativos            
for i in range(0,60):
    if bin_test[i]==bin_modelo2[i]:
        if bin_modelo2[i]==0:
            verd_positivo2 = verd_positivo2+1
        else:
            verd_negativo2 = verd_negativo2+1
    else:
        if bin_modelo2[i]==0:
            falso_positivo2 = falso_positivo2+1
        else:
            falso_negativo2 = falso_negativo2+1

print(verd_positivo2, verd_negativo2, falso_positivo2, falso_negativo2)

#K2Score y BicScore
from pgmpy.estimators import K2Score
k2_score2 = K2Score(data=datos_train).score(modelo2)
print(k2_score2)

from pgmpy.estimators import BicScore
bic_score2 = BicScore(data=datos_train).score(modelo2)
print(bic_score2)



