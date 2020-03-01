import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labenc_x = LabelEncoder()
X[:, 3]=labenc_x.fit_transform(X[:,3])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

#Avoidind dummy variable trap
X=X[:,1:]

#Splitting test and train set
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting simple linear regression to train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Prediction of Test set result
Y_pred=regressor.predict(X_test)


#Buildong backward propogation
import statsmodels.formula.api as sm
X=np.append(arr= np.ones((50,1)).astype(int), values= X , axis = 1)


