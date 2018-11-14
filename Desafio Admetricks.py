
# coding: utf-8

# In[1]:

import numpy as np  
import pandas as pd 
import xlrd

#############################Preparacion de la data#######################################
def Representsfloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def matriz_sum_col(list1,list2):
    listx = []
    for i in range(len(list1)):
        list_aux = []
        for j in range(len(list1[0])):
            list_aux.append(list1[i][j])
        list_aux.append(list2[i])
        listx.append(list_aux)
    return listx

def list_sum_col(list1,list2):
    listx = []
    for i in range(len(list1)):
        listx.append([list1[i],list2[i]])
    return listx
        
    
def string_to_int(lists):
    if Representsfloat(lists[0]) == True:
        for item,value in enumerate(lists,1):
            lists[item - 1] = float(lists[item - 1])
        return lists
    else:
        lists_aux1 = lists
        lists = LabelEncoder().fit_transform(lists)
        lists_aux2 = lists
        return lists,save_code(lists_aux1,lists_aux2)

def save_code(lists,code):
    dic = {}
    for i in range(len(lists)):
        dic[lists[i]] = code[i]
    return dic

def excel_to_dic(name_file):
    workbook = xlrd.open_workbook(name_file)
    workbook = xlrd.open_workbook(name_file, on_demand = True)
    worksheet = workbook.sheet_by_index(0)
    first_row = [] 
    for col in range(worksheet.ncols):
        first_row.append( worksheet.cell_value(0,col) )
    elm = {}
    for col in range(worksheet.ncols):
        data_aux = []
        for row in range(1, worksheet.nrows):
            data_aux.append(worksheet.cell_value(row,col))
        elm[first_row[col]] = data_aux
    return elm

#archivo del mercado del pais 2 meses
real_data = excel_to_dic('admetricks_market_report.xlsx')


# In[3]:

#librerias para los modelos a entrenar
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split 

def init_data(data, variables):
    data_value = []
    if len(variables) == 1:
        code = []
        data_value = data[variables[0]]
        if type(data_value[0])==unicode:
            data_value,code = string_to_int(data_value)
        data_value = np.array(data_value)
    else:
        data_value_aux = []
        for i in range(len(variables)):
            if i==1:
                data_value_aux = list_sum_col(data[variables[i-1]],data[variables[i]])
            if i>1:
                data_value_aux = matriz_sum_col(data_value_aux,data[variables[i]])
        data_value = data_value_aux
        data_value_aux1 = list(zip(*data_value))
        size = len(data_value_aux1)
        code = []
        for i in range(0,size):
            if type(data_value_aux1[i][0])== unicode :
                data_value_aux1[i],code_aux = string_to_int(data_value_aux1[i])
                code.append(code_aux)
        data_value = list(zip(*data_value_aux1))
        data_value_aux1 = []
    return data_value,code
#############################################################################################

#archivo del mercado del pais 2 meses
real_data = excel_to_dic('admetricks_market_report.xlsx')

#variables para el entrenamiento
variables_X = ['Date',
               'Industry',
               'Brand',
             # 'Campaign Name',
               'Campaign Landing Page',
               'Website',
               'Ad Type',
             #  'Ad Size',
             #  'Duration (Video)',
             #  'Skip (Video)',
               'Country',
               'Device',
               'Hosted by',
               'Sold by (Beta)', #sold by(beta)
               'Impact',
               'Impressions']

variables_Y = ['Valuation']

#inicio de los datos en array para ser procesados
X,code_X = init_data(real_data,variables_X)
Y,code_Y = init_data(real_data,variables_Y)

#Separacion del data para test y para train (30% test)
validation_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size)


# In[59]:

from sklearn.ensemble import RandomForestRegressor
import pickle

#Entrenamiento
model = RandomForestRegressor(n_jobs=-2)
model.set_params(n_estimators=30)
model.fit(X_train, Y_train)
print model.score(X_test, Y_test) #score del entrenamiento
filename = 'finalized_model_forest.sav'
pickle.dump(model, open(filename, 'wb'))


# In[60]:

#Funcion para codificar con los mismos numeros usados para el entrenamiento
def same_code(code,data):
    list_aux = []
    data_value = []
    new = False
    if Representsfloat(data[0]) == True:
        for item,value in enumerate(data,1):
            data[item - 1] = float(data[item - 1])
        return data
    else:
        new = False
        for i in range(len(data)):
            if data[i] in code.keys():
                data_value.append(code[data[i]])
            else:
                new = True
                list_aux.append(data[i])

        if new:
            lists_aux1 = list_aux
            list_aux = LabelEncoder().fit_transform(list_aux)
            lists_aux2 = list_aux
            new_code = save_code(lists_aux1,lists_aux2)
            code.update(new_code)
            for i in range(len(data)):
                if data[i] in code.keys():
                    data_value.append(code[data[i]])
        return data_value
    
#iniciar data con la misma codificacion usada para entrenar el modelo
def init_data1(data, variables, code):
    data_value = []
    if len(variables) == 1:
        data_value = data[variables[0]]
        #print type(data_value[0])
        if type(data_value[0])==unicode:
            data_value = same_code(code,data_value)
        data_value = np.array(data_value)
    else:
        data_value_aux = []
        for i in range(len(variables)):
            if i==1:
                data_value_aux = list_sum_col(data[variables[i-1]],data[variables[i]])
            if i>1:
                data_value_aux = matriz_sum_col(data_value_aux,data[variables[i]])
        data_value = data_value_aux
        data_value_aux1 = list(zip(*data_value))
        size = len(data_value_aux1)
        for i in range(0,size):
            if type(data_value_aux1[i][0])== unicode :
                data_value_aux1[i] = same_code(code[i],data_value_aux1[i])
        data_value = list(zip(*data_value_aux1))
        data_value_aux1 = []
    return data_value

#archivo del Banco Estado 7 meses aprox
muestral_data = excel_to_dic('admetricks_brand_report.xlsx')

variables_X = ['Date',
               'Industry',
               'Brand',
             # 'Campaign Name',
               'Campaign Landing Page',
               'Website',
               'Ad Type',
             #  'Ad Size',
             #  'Duration (Video)',
             #  'Skip (Video)',
               'Country',
               'Device',
               'Hosted by',
               'Sold by', #sold by(beta)
               'Impact',
               'Impressions']

variables_Y = ['Valuation']

#inicio de los datos cargando la codificacion usada en el entrenamiento
X = init_data1(muestral_data,variables_X,code_X)
Y = init_data1(muestral_data,variables_Y,code_Y)

Y_pred = model.predict(X)
print 'El banco estado invirtio un total de: ',sum(Y_pred) 


# In[57]:

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


dates = []
for value in muestral_data['Date']:
    dates.append(datetime.strptime(value, '%Y-%m-%d'))

#grafico modelo ML vs modelo admetricks
p1 = plt.plot(dates,Y_pred,"--",color = "red", label = 'Random forest model')
p2 = plt.plot(dates,Y,color = "blue",label = 'Metodo Admetricks')
plt.ylabel('Valorizaciones [$]',fontsize = 21)
plt.legend()
plt.rcParams["figure.figsize"] = 30,16
plt.rcParams["xtick.labelsize"] = "22"
plt.rcParams["ytick.labelsize"] = "22"
plt.rcParams["legend.fontsize"] = "23"
plt.show()



# In[ ]:



