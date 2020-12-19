#!/usr/bin/env python
# coding: utf-8

# ## Library and Data Loading

# In[1]:


#Import libraries

import pandas as pd
import numpy as np
pd.options.display.max_columns = None
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats

pd.options.display.max_columns = None

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
sns.set(rc={'figure.figsize':(12, 8)})

import statsmodels.api as sm

#Load data
bike = pd.read_csv('SeoulBikeData.csv',encoding='ISO-8859-1')


# ## Data Preparation

# In[29]:


#Load data
bike = pd.read_csv('SeoulBikeData.csv',encoding='ISO-8859-1')

#Nonfunctioning days do not have a system to log rental activity. Drop all nonfunctioning day records.
indexNames = bike[bike['Functioning Day'] == 'No' ].index
bike.drop(indexNames , inplace=True)

#Transform Date column into Date type
bike['Date'] = bike['Date'].astype('datetime64[ns]')

#Rename Holiday to Holiday1

bike.rename(columns={'Holiday':'Holiday_'}, inplace=True)

#Get dummy variables
dummy_variable_1 = pd.get_dummies(bike["Seasons"])
bike = pd.concat([bike, dummy_variable_1], axis=1)

dummy_variable_2 = pd.get_dummies(bike["Holiday_"])
bike = pd.concat([bike, dummy_variable_2], axis=1)

dummy_variable_3 = pd.get_dummies(bike["Functioning Day"])
dummy_variable_3.rename(columns={'Yes':'Functioning Day', 'No':'Nonfunctioning Day'}, inplace=True)
bike = pd.concat([bike, dummy_variable_3], axis=1)

#Create columns for Month

bike['Month'] = bike['Date'].dt.month

#OneHot months

dummy_variable_4 = pd.get_dummies(bike["Month"])
dummy_variable_4.rename(columns={1:'January', 2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}, inplace=True)
bike = pd.concat([bike, dummy_variable_4], axis=1)

#OneHot Hours
dummy_variable_5 = pd.get_dummies(bike["Hour"])
bike = pd.concat([bike, dummy_variable_5], axis=1)

#Add Weekday column 
bike['Weekday'] = bike['Date'].dt.dayofweek

#Change names of the weekdays from numbers to actual names
conditions1 = [
    (bike['Weekday'] == 0),
    (bike['Weekday'] == 1) ,
    (bike['Weekday'] ==2) ,
    (bike['Weekday'] == 3) ,
    (bike['Weekday'] == 4),
    (bike['Weekday'] == 5),
    (bike['Weekday'] == 6)
    ]

## create a list of the values we want to assign for each condition
values1 = ['Sunday', 'Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday']

## create a new column and use np.select to assign values to it using our lists as arguments
bike['WeekDay'] = np.select(conditions1, values1)






####################
#Change names of the weekdays from numbers to actual names
conditions24 = [
    (bike['Hour'] <= 5),
    (bike['Hour'] > 5)  & (bike['Hour'] <= 10 )  ,
    (bike['Hour'] > 10) & (bike['Hour'] <= 15) ,
    (bike['Hour'] > 15) & (bike['Hour'] <= 22 ) ,
    (bike['Hour'] > 22)
    ]

## create a list of the values we want to assign for each condition
values24 = ['LateNight', 'Morning', 'Midday', 'Evening','LateNight']

## create a new column and use np.select to assign values to it using our lists as arguments
bike['TimeDay'] = np.select(conditions24, values24)






#Create a binary for variable for raining conditions. Is it raining? (yes/no)
conditions2 = [
    (bike['Rainfall(mm)'] == 0),
    (bike['Rainfall(mm)'] > 0) 
        ]

## create a list of the values we want to assign for each condition
values2 = [0, 1]

## create a new column and use np.select to assign values to it using our lists as arguments
bike['Is_Raining'] = np.select(conditions2, values2)


#Create a binary for variable for snowing conditions. Is it snowing? (yes/no)
conditions3 = [
    (bike['Snowfall (cm)'] == 0),
    (bike['Snowfall (cm)'] > 0) 
        ]

## create a list of the values we want to assign for each condition
values3 = [0, 1]

## create a new column and use np.select to assign values to it using our lists as arguments
bike['Is_Snowing'] = np.select(conditions3, values3)



bike.head()


# ## Data Exploration (Type)

# In[3]:


bike.shape


# In[4]:


bike.describe()


# In[5]:


bike.isnull().sum()


# In[6]:


bike.dtypes


# In[7]:


bike.info()


# ## Data Exploration (Visual)

# **Time vs Rented Bikes**

# In[30]:


sns.boxplot(x=bike["Seasons"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Seasons")
plt.show()

sns.boxplot(x=bike["Hour"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Hour")
plt.show()

sns.boxplot(x=bike["WeekDay"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Weekday")
plt.show()

sns.boxplot(x=bike["TimeDay"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Time of the day")
plt.show()


# **Rented Bike by Holiday**

# In[9]:



sns.boxplot(x=bike["Holiday"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Holiday")
plt.show()


# **Weather Conditions (Humidity and Temperature) vs Rented Bikes**

# In[10]:



sns.regplot(x=bike['Temperature(°C)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Temperature(°C) and Rented Bike Count")
plt.show()

sns.regplot(x=bike['Humidity(%)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Humidity(%) and Rented Bike Count")
plt.show()

sns.regplot(x=bike['Dew point temperature(°C)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Visibility (10m) and Rented Bike Count")
plt.show()


# **Weather Conditions (Miscellaneous) vs Rented Bikes**

# In[11]:


sns.regplot(x=bike['Wind speed (m/s)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Wind speed (m/s) and Rented Bike Count")
plt.show()

sns.regplot(x=bike['Visibility (10m)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Visibility (10m) and Rented Bike Count")
plt.show()

sns.regplot(x=bike['Solar Radiation (MJ/m2)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Visibility (10m) and Rented Bike Count")
plt.show()


# **Weather conditions correlation with Rented Bikes Count**

# In[12]:


coef, p_val = stats.pearsonr(bike['Temperature(°C)'], bike['Rented Bike Count'])
print('Correlation Between Rented Bike Count and Temperature, Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))

coef, p_val = stats.pearsonr(bike['Humidity(%)'], bike['Rented Bike Count'])
print('\nCorrelation Between Rented Bike Count and Humidity(%), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))

coef, p_val = stats.pearsonr(bike['Wind speed (m/s)'], bike['Rented Bike Count'])
print('\nCorrelation Between Rented Bike Count and Wind speed (m/s), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))

coef, p_val = stats.pearsonr(bike['Visibility (10m)'], bike['Rented Bike Count'])
print('\nCorrelation Between Rented Bike Count and Visibility (10m), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))

coef, p_val = stats.pearsonr(bike['Dew point temperature(°C)'], bike['Rented Bike Count'])
print('\nCorrelation Between Rented Bike Count and Dew point temperature(°C), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))

coef, p_val = stats.pearsonr(bike['Solar Radiation (MJ/m2)'], bike['Rented Bike Count'])
print('\nCorrelation Between Rented Bike Count and Solar Radiation (MJ/m2), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))


# **Precipitation vs Rented Bikes**

# In[13]:



sns.regplot(x=bike['Rainfall(mm)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Rainfall(mm) and Rented Bike Count")
plt.show()

sns.regplot(x=bike['Snowfall (cm)'], y=bike['Rented Bike Count'], line_kws={"color": "red"}).set_title("Scatterplot of Snowfall (cm) and Rented Bike Count")
plt.show()


# **Precipitation (binary) vs Rented Bikes**

# In[14]:


sns.boxplot(x=bike["Is_Raining"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Raining?")
plt.show()

sns.boxplot(x=bike["Is_Snowing"], y=bike["Rented Bike Count"])
plt.title("Boxplot of Rented Bike Count, grouped by Snowing?")
plt.show()


# **Precipitation correlation with Rented Bike Count**

# In[15]:


coef, p_val = stats.pearsonr(bike['Rainfall(mm)'], bike['Rented Bike Count'])
print('Correlation Between Rented Bike Count and Rainfall(mm), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))

coef, p_val = stats.pearsonr(bike['Snowfall (cm)'], bike['Rented Bike Count'])
print('\nCorrelation Between Rented Bike Count and Snowfall (cm), Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))


# **Correlation between Dew Point Temperature and Ambient Temperature**

# In[16]:


sns.regplot(x=bike['Temperature(°C)'], y=bike['Dew point temperature(°C)'], line_kws={"color": "red"}).set_title("Scatterplot of Temperature and Dew Point")
plt.show()

coef, p_val = stats.pearsonr(bike['Temperature(°C)'], bike['Dew point temperature(°C)'])
print('Correlation Between Temperature and Dew Point, Coeff: {:.4f}, p value: {:.4f}'.format(coef, p_val))


# ## Variables to drop##
# `Visibility (10m)`: Apparent low correlation
# `Dew point temperature(°C)`: Strongly correlated to `Temperature(°C)`. Favoured `Temperature(°C)` varaible over it. 
# `Solar Radiation (MJ/m2)`: Apparent low correlation
# `Wind speed (m/s)`: Apparent low correlation

# **VIF Analysis**

# In[17]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Set up X and Y
X_columns = ['Temperature(°C)','Humidity(%)','Rainfall(mm)','Snowfall (cm)','Summer','Winter','Spring', 'Holiday',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
X = bike[X_columns]
Y = bike['Rented Bike Count']
X = sm.add_constant(X)

# Calculate VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Inspect VIF Values, round it to 2 decimal places
vif.round(2)


# **Regression Model**

# In[18]:


X_columns = ['Temperature(°C)','Rainfall(mm)','Snowfall (cm)','Summer','Winter','Spring', 'Holiday',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
X = bike[X_columns]
Y = bike['Rented Bike Count']
X = sm.add_constant(X)
model2 = sm.OLS(Y, X).fit()
print(model2.summary())


# **Partial Regression Plots for all Predictors**

# In[19]:


fig = plt.figure(figsize=(20,100))
fig = sm.graphics.plot_partregress_grid(model2, fig=fig)
plt.show()


# **Distribution of the Residuals around 0** 

# In[20]:


sns.distplot(model2.resid)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')
plt.show()


# **Residuals vs. Predicted Values**

# In[21]:


sns.scatterplot(model2.fittedvalues, model2.resid)
plt.title('Plot of Residuals vs. Fitted Values')
plt.xlabel('Fitted value')
plt.ylabel('Residuals')
plt.show()


# **Cook's Distance**

# In[22]:


influence = model2.get_influence()
# show the influence analysis summary
print(influence.summary_table())


# **Cook's Distance (Visual)**

# In[23]:


(c, p) = influence.cooks_distance
plt.stem(np.arange(len(c)), c, markerfmt=",", use_line_collection=True)
plt.show()


# ## Interpretation of Seoul Bike Sharing Demand Model
# 
# 1. **Multiple regression equation**
# 
# `Rented_Bike_Count` = 488.5 + 22.9691 `Temperature(°C)` - 96.104 `Rainfall(mm)` - 30.07 `Snowfall `(cm) - 173.6773 `Summer` - 319.7467 `Winter` - 155.1481 `Spring` - 108.6099 `Holiday` - 114.2619 * `1:00` - 232.4753 * `2:00` - 322.9591 * `3:00` - 389.4691 * `4:00` - 384.8214 * `5:00` - 218.9294 * `6:00` + 108.3821 * `7:00` + 520.834 * `8:00` + 123.5212 * `9:00` - 47.7561 * `10:00` - 1.2803 * `11:00` + 75.4693 * `12:00` + 96.1594 * `13:00` + 93.4536 * `14:00` + 166.5969 * `15:00` + 266.8455 * `16:00` + 502.9557 * `17:00` + 915.945 * `18:00` + 606.0023 * `19:00` + 510.2418 * `20:00` + 485.4715 * `21:00` + 369.1542 * `22:00` + 120.3063 * `23:00`
# 
# 
# 2. **Overall goodness-of-fit: R-square, F-value**
# 
# 
# - R-square: 62% of variation in `Rented_Bike_Count` is explained by the regression model with thirty variables Temperature(°C),Rainfall(mm),Snowfall (cm), Summer, Winter, Spring, Holiday, 1:00, 2:00, 3:00, 4:00, 5:00, 6:00, 7:00, 8:00, 9:00, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 16:00, 17:00, 18:00, 19:00, 20:00, 21:00, 22:00 and 23:00
# 
# - F-test: The p-value of F-test is less than 0.001, indicating that at least one predictor in the model has a non-zero regression coefficient. 
# 
# 
# 3. **Assessment of individual predictiors**
#  
# Weather Predictors  
# - `Temperature(°C)` positively affects `Rented Bike Count`, b = 22.9691, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. Each additional degree Celsius will result in 22.9691 more bikes being rented.
# - `Rainfall(mm)` negatively affects `Rented Bike Count`, b = -96.104, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. Each additional mm of rain will result in 96.104 less bikes being rented.
# - `Snowfall (cm)` negatively affects `Rented Bike Count`, b = -30.07, p < 0.05 indicating that this variable is significantly related to `Rented Bike Count`. Each additional cm of snowfall will result in 30.07 less bikes being rented
# 
# Season Predictors
# - `Summer` negatively affects `Rented Bike Count`, b = -173.6773, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. During Summer, 173.6773 less bikes will be rented than during Autumn.
# - `Winter` negatively affects `Rented Bike Count`, b = -319.7467, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. During Summer, 319.7467 less bikes will be rented than during Autumn.
# - `Spring` negatively affects `Rented Bike Count`, b = -155.1481, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. During Summer, 155.1481 less bikes will be rented than during Autumn.
# 
# Holiday Predictor
# - `Holiday` negatively affects `Rented Bike Count`, b = -108.6099, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. During a Holiday, 108.6099 less bikes will be rented that during a non Holiday.
# 
# Hourly Predictor
# - `1:00` negatively affects `Rented Bike Count`, b = -114.2619, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 1:00, 114.2619 less bikes will be rented than at 0:00.
# - `2:00` negatively affects `Rented Bike Count`, b = -232.4753, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 2:00, 232.4753 less bikes will be rented than at 0:00.
# - `3:00` negatively affects `Rented Bike Count`, b = -322.9591, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 3:00, 322.9591 less bikes will be rented than at 0:00.
# - `4:00` negatively affects `Rented Bike Count`, b = -389.4691, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 4:00, 389.4691 less bikes will be rented than at 0:00.
# - `5:00` negatively affects `Rented Bike Count`, b = -384.8214, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 5:00, 384.8214 less bikes will be rented than at 0:00.
# - `6:00` negatively affects `Rented Bike Count`, b = -218.9294, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 6:00, 218.9294 less bikes will be rented than at 0:00.
# - `7:00` positively affects `Rented Bike Count`, b = 108.3821, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 7:00, 108.3821 more bikes will be rented than at 0:00.
# - `8:00` positively affects `Rented Bike Count`, b = 520.834, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 8:00, 520.834 more bikes will be rented than at 0:00.
# - `9:00` positively affects `Rented Bike Count`, b = 123.5212, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 9:00, 123.5212 more bikes will be rented than at 0:00.
# - `10:00` is not a significant predictor for`Rented Bike Count`, p = 0.11,  indicating that the cofficient of -47.7561 is not significantly different from zero. Therefore, holding other variables constant, there is no statistical difference in `Rented Bike Count`when renting bikes at `10:00` 
# - `11:00` is not a significant predictor for`Rented Bike Count`, p = 0.966,  indicating that the cofficient of -1.2803 is not significantly different from zero. Therefore, holding other variables constant, there is no statistical difference in `Rented Bike Count`when renting bikes at `11:00` 
# - `12:00` positively affects `Rented Bike Count`, b = 75.4693, p < 0.05 indicating that this variable is significantly related to `Rented Bike Count`. At 12:00, 72.4693 more bikes will be rented than at 0:00.
# - `13:00` positively affects `Rented Bike Count`, b = 96.1594, p < 0.05 indicating that this variable is significantly related to `Rented Bike Count`. At 13:00, 96.1594 more bikes will be rented than at 0:00.
# - `14:00` positively affects `Rented Bike Count`, b = 93.4536, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 14:00, 93.4536 more bikes will be rented than at 0:00.
# - `15:00` positively affects `Rented Bike Count`, b = 166.5969, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 15:00, 166.5969 more bikes will be rented than at 0:00.
# - `16:00` positively affects `Rented Bike Count`, b = 266.8455, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 16:00, 266.8455 more bikes will be rented than at 0:00.
# - `17:00` positively affects `Rented Bike Count`, b = 502.9557, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 17:00, 502.9557 more bikes will be rented than at 0:00.
# - `18:00` positively affects `Rented Bike Count`, b = 915.945, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 18:00, 915.945 more bikes bikes will be rented than at 0:00.
# - `19:00` positively affects `Rented Bike Count`, b = 606.0023, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 19:00, 606.0023 more bikes will be rented than at 0:00.
# - `20:00` positively affects `Rented Bike Count`, b = 510.2418, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 20:00, 510.2418 more bikes will be rented than at 0:00.
# - `21:00` positively affects `Rented Bike Count`, b = 485.4715, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 21:00, 485.4715 more bikes will be rented than at 0:00.
# - `22:00` positively affects `Rented Bike Count`, b = 369.1542, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 22:00, 389.1542 more bikes will be rented than at 0:00.
# - `23:00` positively affects `Rented Bike Count`, b = 120.3063, p < 0.001 indicating that this variable is significantly related to `Rented Bike Count`. At 23:00, 120.3063 more bikes will be rented than at 0:00.
# 
# 
# 
# 4. **Model Assumptions:** Residuals are nomally distributed around zero. The residual vs. predicted value plot shows no autocorrelation. The variance of resisuals is also approximatley constant. 
# 
# 
# 5. **Influential observations:** All Cook's D scores are less than 1. No outlier or influential observation was observed. Residual vs Fitted plot shows a pattern. Further investigation of variables needs to happen.
# 
# 
# 6. **No severe multicollinearity**: As expected, `Temperature`, `Winter` and `Summer`  are somewhat correlated in this dataset. But the VIF values for all predictors are smaller than 10, so multicollinearity is not severe. 

# In[ ]:




