#!/usr/bin/env python
# coding: utf-8

# # Task 2 - To Explore Supervised Machine Learning
# ## In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# Task - What will be predicted score if a student study for 9.25 hrs in a
# day?

# In[164]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[189]:


# Importing file to do regression analysis
data = pd.read_csv('http://bit.ly/w-data')
data.tail()


# ### We do a quick analysis to check that the data we have are are dependent on each other, the value of r-squared we have here suggests the Y(Scores) is heavily dependent on X(Hours of study).

# In[178]:


x1=sm.add_constant(x)
reg=sm.OLS(y,x1).fit()
reg.summary()


# ## Plotting chart to see that is the data suitable for regression analysis.

# In[179]:


#Plotting on scatter chart
x = data['Hours'].values
y = data['Scores'].values
plt.scatter(x,y)
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.title('Study of Scores & study hours')
plt.show()


# In[180]:


# Reshaping array to 2D
x=x.reshape(-1,1)
y=y.reshape(-1,1)


# ## Plotting the regression line

# In[181]:


print("Y-Intercept=",lr.intercept_)
print("Slope=",lr.coef_)
#plotting regression line
line = lr.coef_*x+lr.intercept_
plt.scatter(x,y)
plt.ylabel('Scores')
plt.xlabel('Hours')
plt.title('Study of Scores & study hours')
plt.plot(x,line,c='Black')
plt.show()


# ### Divides 80% data for training, 20% for testing

# In[182]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[183]:


x_train


# ## To train the model

# In[184]:


lr.fit(x_train,y_train)


# ## Task - What will be predicted score if a student study for 9.25 hrs in a day -

# In[185]:


hours=[[9.25]]
pred_h=lr.predict(hours)
print('If a student studies 9.25 hours in a day, he will score:', pred_h)


# In[ ]:




