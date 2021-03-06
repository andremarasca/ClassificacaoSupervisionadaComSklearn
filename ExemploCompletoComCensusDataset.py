﻿import numpy as np
import pandas as pd
import copy

#%% Definir o modelo de aprendizagem que será utilizado

modelo_desejado = 'Redes Neurais'
#modelo_desejado = 'Arvores de Decisão'
#modelo_desejado = 'KNN'
#modelo_desejado = 'Naive Bayes'

if modelo_desejado == 'Redes Neurais':
    from sklearn.neural_network import MLPClassifier
    classificador = MLPClassifier(max_iter = 1000, tol = 0.0001)
elif modelo_desejado == 'Arvores de Decisão':
    from sklearn.tree import DecisionTreeClassifier
    classificador = DecisionTreeClassifier(criterion = 'entropy')
elif modelo_desejado == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier
    classificador = KNeighborsClassifier(n_neighbors=3)
elif modelo_desejado == 'Naive Bayes':
    from sklearn.naive_bayes import GaussianNB
    classificador = GaussianNB()

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

#%%#########################################
### Validacao Estatistica dos resultados ###

# Instanciar objeto para normalização Z-score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.metrics import confusion_matrix

# Descobrindo o número de instâncias da base
n_inst = base.shape[0]

# K-fold implementado na mão,
# os dados devem ser ordenados pela classe
n_folds = 2
kfold = []
for i in range(n_inst):
      kfold.append(i % n_folds)
      
# Loop da Validação cruzada K-fold
Rates = []
for pasta in range(n_folds):
      # A cada iteração deve-se copiar os previsores novamente
      # Pois a normalização z-score altera os dados
      previsoresValidacao = copy.deepcopy(previsores)
      
      # Descobre quais instâncias estão na pasta de treino
      # E quais instâncias estão na pasta de teste
      train = []
      test = []
      for i in range(n_inst):
            if kfold[i] == pasta:
                  test.append(i)
            else:
                  train.append(i)
      
      # A cada iteração a z-score normaliza TODOS os dados, para isso ela obtém a
      # média e o desvio padrão dos dados de TREINAMENTO (MUITO IMPORTANTE)
      scaler_previsores = scaler.fit(previsoresValidacao[train])
      previsoresValidacao = scaler_previsores.transform(previsoresValidacao)
      
      # Treinamento do classificador com as instâncias de treino
      classificador.fit(previsoresValidacao[train], classe[train])
      # Predição dos rótulos das instâncias de teste
      classe_predita = classificador.predict(previsoresValidacao[test])
      # Compara os rótulos preditos com os rótulos reais
      MC = confusion_matrix(classe[test], classe_predita)
      Rate = MC.diagonal().sum() / MC.sum()
      print(MC)
      print(Rate)
      Rates.append(100*Rate)

# Calcula média e desvio padrão das taxas de sucesso do K-fold
media = np.mean(Rates)
desvio = np.std(Rates)
print('media = %.2f %%, Desvio = %.2f %%' %(media,desvio))

### Fim da Validacao Estatistica dos resultados ###
###################################################