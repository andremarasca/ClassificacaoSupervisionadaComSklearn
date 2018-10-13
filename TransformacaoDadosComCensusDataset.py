import numpy as np
import pandas as pd

#%% Leitura dos dados

# Utilizar o pandas para ler o dataset
base = pd.read_csv('census.csv')

# Ordenar dados pela classe
base = base.sort_values(by=base.keys()[-1])

#%% Pre-processamento nos atributos

# Definir os indices de cada tipo de atributo
# Os indices que nao apareceram foi porque foram considerados
# inuteis para classificacao
numericos = [0, 10, 11, 12]
nominais = [1, 5, 6, 7, 8, 9, 13]
ordinais = [3]

tam = base.shape[0]

atrib_num = base.iloc[:, numericos].values
atrib_nom = base.iloc[:, nominais].values
atrib_ord = base.iloc[:, ordinais].values
classe = base.iloc[:, 14].values

# Transformar a classe em numerico
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

encoder_classe = labelencoder.fit(classe)
classe = encoder_classe.transform(classe)

# conversao dos atributos ordinais em numericos

educacao = {' 10th' : 0, ' 11th' : 0, \
            ' 12th' : 0, ' 1st-4th' : 0, \
            ' 5th-6th' : 0, ' 7th-8th' : 0, \
            ' 9th' : 0, ' Assoc-acdm' : 7, \
            ' Assoc-voc' : 8, ' Bachelors' : 3, \
            ' Doctorate' : 6, ' HS-grad' : 1, \
            ' Masters' : 5, ' Preschool' : 0, \
            ' Prof-school' : 4, ' Some-college' : 2};

for i in range(tam):
      if atrib_ord[i][0] in educacao:
            atrib_ord[i][0] = educacao[atrib_ord[i][0]]
            
atrib_ord = atrib_ord.astype(int)

# Convercao de atributos nominais para numericos

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

classe = labelencoder.fit_transform(classe)

for i in range(len(nominais)):
      atrib_nom[:,i] = labelencoder.fit_transform(atrib_nom[:,i])

atrib_nom = atrib_nom.astype(int)

# Conversao dos nominais que agora sao inteiros em binarios

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

atrib_nom = onehotencoder.fit_transform(atrib_nom).toarray()

# Concatenar atributos gerando DataSet
previsores = np.concatenate((atrib_num, atrib_ord, atrib_nom), axis = 1)