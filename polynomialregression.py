
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2:].values
X.shape
Y.shape

#splitting the data set into dataset and training set
"""from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)"""


#feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

#splitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#fitting polynomial Regression too the dataset
from sklearn.preprocessing import PolynomialFeatures 
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,Y)

#Visualising the  linear Regression results

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("True of Bluff(Linear Regression)")
plt.xlabel("PositionLevel")
plt.ylabel("Salary")
plt.show()

#Visualising Polynomial Regressio n Results
#X_grid=np.arange(min(X),0.1)
#X_grid=X_grid((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("True of Bluff(Polynomial Regression)")
plt.xlabel("PositionLevel")
plt.ylabel("Salary")
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)

lin_reg_2.predict(poly_reg.fit_transform(6.5))