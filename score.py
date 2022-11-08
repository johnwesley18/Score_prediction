#!/usr/bin/env python
# coding: utf-8

# In[2]:


import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot


# In[4]:


data = pd.read_csv('score.csv')
data


# In[4]:


data = pd.read_csv('score.csv')
fig = px.scatter(data, x='overs', y='score')
fig.show()


# In[5]:


formula = LinearRegression()
x = data.overs.values.reshape(-1, 1)
y = data.score.values.reshape(-1, 1)
x


# In[7]:


formula.fit(x, y)
LinearRegression()
predict_score = formula.predict([[20]])
print(predict_score)


# In[ ]:
