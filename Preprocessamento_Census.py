import pandas as pd

# IMPORTAR BASE DO CENSUS
base = pd.read_csv('census.csv')

tam = base.shape[0]

# lembrar de remover peso final (indice 2)
# lembrar de remover num educacao(indice 4)

numericos = [0, 10, 11, 12]
nominais = [1, 5, 6, 7, 8, 13]
ordinais = [3, 9]

atrib_num = base.iloc[:, numericos].values
atrib_nom = base.iloc[:, nominais].values
atrib_ord = base.iloc[:, ordinais].values
classe = base.iloc[:, 14].values

# conversao dos atributos ordinais em numericos

import numpy as np

educacao = {' 10th' : 0, ' 11th' : 0, \
            ' 12th' : 0, ' 1st-4th' : 0, \
            ' 5th-6th' : 0, ' 7th-8th' : 0, \
            ' 9th' : 0, ' Assoc-acdm' : 7, \
            ' Assoc-voc' : 8, ' Bachelors' : 3, \
            ' Doctorate' : 6, ' HS-grad' : 1, \
            ' Masters' : 5, ' Preschool' : 0, \
            ' Prof-school' : 4, ' Some-college' : 2};

sexo = {' Female' : 0, ' Male' : 1}

for i in range(tam):
      if atrib_ord[i][0] in educacao:
            atrib_ord[i][0] = educacao[atrib_ord[i][0]]
      if atrib_ord[i][1] in sexo:
            atrib_ord[i][1] = sexo[atrib_ord[i][1]]
            
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

# Normalizacao dos atributos numericos

# Normalizacao z-score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
atrib_num = scaler.fit_transform(atrib_num)

previsores = np.concatenate((atrib_num, atrib_ord, atrib_nom), axis = 1)

####### Aqui acabou o preprocessamento

# n_splits = numero de pastas
# shuffle = selecao dos indices para as pastas aleatorio
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, shuffle = True)
X = range(0, len(classe))

from sklearn.neural_network import MLPClassifier
rede = MLPClassifier(verbose = False, max_iter = 1000, tol = 0.0001)

from sklearn.metrics import confusion_matrix

Rates = []

for train, test in kf.split(X):
      rede.fit(previsores[train], classe[train])
      classe_predita = rede.predict(previsores[test])
      MC = confusion_matrix(classe[test], classe_predita, labels = [0, 1])
      acuracia = MC.diagonal().sum() / MC.sum()
      print(MC)
      print(acuracia)
      Rates.append(acuracia)

import numpy as np
media = np.mean(Rates)
desvio = np.std(Rates)

print('Media = %f' %(media*100))
print('Desvio = %f' %(desvio*100))







