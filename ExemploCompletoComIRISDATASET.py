import numpy as np
import pandas as pd
import copy

#%% Definir o modelo de aprendizagem que será utilizado

#modelo_desejado = 'Redes Neurais'
#modelo_desejado = 'Arvores de Decisão'
#modelo_desejado = 'KNN'
modelo_desejado = 'Naive Bayes'

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

#%% Leitura dos dados e preprocessamento

# Utilizar o pandas para ler o dataset
base = pd.read_csv('IRIS_DATASET.csv')

# Ordenar dados pela classe
base = base.sort_values(by=base.keys()[-1])

# Separar as atributos previsores da classe
classe = base.pop(base.keys()[-1])
previsores = base.values

# Transformar a classe em numerico
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

encoder_classe = labelencoder.fit(classe)
classe = encoder_classe.transform(classe)


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
n_folds = 10
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

#%%##################################################
### Teste de mesa sobre o classificador treinado ###

def testar_classificacao_na_mao(previsores, classe):

    
    # Normalizacao Z-score
    scaler_previsores = scaler.fit(previsores)
    previsores = scaler_previsores.transform(previsores)
    
    # Treinar a base com todos os dados disponiveis para testar entradas na mao
    classificador.fit(previsores, classe)
    
    while True:
          # Precicao dos d ados de entrada
          A = input('Informe o comprimento da sepala em cm: ')
          B = input('Informe a largura da sepala em cm: ')
          C = input('Informe o comprimento da petala em cm: ')
          D = input('Informe a largura da petala em cm: ')
          entrada = np.array([[float(A),float(B),float(C),float(D)]])
          #Z-score da entrada
          entrada = scaler_previsores.transform(entrada)
          # predicao
          classe_predita = classificador.predict(entrada)
          # Descobre o nome da classe
          classe_predita = encoder_classe.inverse_transform(classe_predita)
          # mostrar a saida
          print(classe_predita[0])
          #imprimir as probabilidades
          print(classificador.predict_proba(entrada))
     
# Descomentar o código abaixo caso esteja usando iris dataset e queira testar na mão
#testar_classificacao_na_mao(previsores, classe)
        
### Fim do Teste de mesa sobre o classificador treinado ###
###########################################################


