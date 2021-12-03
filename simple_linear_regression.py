#!/usr/bin/env python
# coding: utf-8

# # Algoritmos básicos de minería de datos o machine learning.

# # Workshop: Regresión Lineal

# Importamos los módulos necesarios para el desarrollo un modelo de regresión lineal.

# In[1]:



get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# Generamos los datos sintéticos para alimentar al modelo de regresión lineal.
# 
# random generator , permite jugar con números para generar números randomicos

# In[2]:



rng = np.random.default_rng()

# generate random data
x = rng.random(14)
y = 1.4*x + rng.random(14)
y


# Calculamos la regresión lineal de mínimos cuadrados para dos conjuntos de datos.

# In[3]:



slope, intercept, r, p, std_err = stats.linregress(x, y)


# Utilizamos la función auxiliar que usa la pendiente e intersección calculadas para devolver un nuevo valor.

# In[4]:



def fitted_value(x):
    return slope * x + intercept


# Calcular los valores ajustados para el eje ‘y’ usando una función auxiliar

# In[5]:



new_values = list(map(fitted_value, x))


# Visualización del modelo.

# In[6]:


# Draw the original scatter plot
plt.plot(x, y, 'o', label='original data')

# Draw the line or linear regression
plt.plot(x, new_values, 'r', label='fitted line')

plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:




