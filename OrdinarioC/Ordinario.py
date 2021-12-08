#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: brayan
"""
import numpy as np
def heaviside(x):
    for i in range(0,len(x)):
        if x[i]>= 0:
            x[i]=1
        else:
            x[i]=-1
    return x

def sigmoid(z):
    return 1/(1+np.exp(-z))

def derivadaSigmoid(z):
    return np.power(np.exp(-z),2.0)/np.power(1+np.exp(-z),2.0)

def Derivadatanh(t):
    return 1 - t**2 

def tanh(t):
    return np.tanh(t) 

x = np.array([0.1,-0.5,0.3,0.2]) 
Y = np.array([1.0,0.0])

w = np.array([[0.1,0.3,0.8,0.3],[0.1,-0.5,-0.8,0.2],[0.3,-0.3,-0.8,-0.5]])
a2 = np.matmul(w,x)
a2[0] +=0.1
a2[1] +=0.1
a2[2] +=0.1
a2_2 = heaviside(a2)
    
#calcular 2 capa oculta 

w1 = np.array([[0.4,-0.5,0.3],[0.3,-0.5,-0.7]])
a3 = np.matmul(w1,a2_2)
a3[0] +=-0.5
a3[1] +=-0.3
a3_3 = tanh(a3)

#calculamos la salida
w2 = np.array([[0.6,-0.1],[0.5,0.4]])
y1=np.matmul(w2,a3_3)
y1[0] += 0.4
y1[1] +=-0.5
y = sigmoid(y1)
print("Primer salida red neuronal: ",y)
softmax = np.exp(y[0]) / (np.exp(y[0]) +np.exp(y[1]))
softmax1 =  np.exp(y[1]) / (np.exp(y[0]) +np.exp(y[1]))
softmaxtotal = softmax + softmax1
print("softmax: ",softmaxtotal)


#******************************************RETROPROPAGACION****************************** 
# pesos sinapticos a actualizar
pesos = np.array([0.6,0.5,-0.1,-0.4])
calculoError = -sum(np.power((Y-y),2.0))
print("error: ",calculoError)
funcionDerivada = Derivadatanh(sum(np.matmul(w2,a3_3)))
j=0
for i in range(0,len(pesos)):
    pesoActualizado = pesos[i] - 0.5*(calculoError*funcionDerivada * a3_3[j])
    pesos[i] = pesoActualizado
    if(i<2):
        j=0
    elif(i>=2 and i<4):
        j=1
        
#******************************************************************************************


# Propagacion hacia adelante
pesos1 =pesos
print("Nuevos pesos sinapticos: ", pesos1)
y1=np.matmul(pesos1.reshape(2,2),a3_3)
y1[0] += 0.4
y1[1] +=-0.5    
y = sigmoid(y1)
print("nueva salida red neuronal: ",y)
softmax = np.exp(y[0]) / (np.exp(y[0]) +np.exp(y[1]))
softmax1 =  np.exp(y[1]) / (np.exp(y[0]) +np.exp(y[1]))
softmaxtotal = softmax + softmax1
print("softmax: ",softmaxtotal)



calculoError = -sum(np.power((Y-y),2.0))
print("error: ",calculoError)