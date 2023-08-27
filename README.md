# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Agalya R 
RegisterNumber:212222040003

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
*/
```


## Output:
![1](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/f5dffe8d-5a8c-42f7-a1b5-7e1afdaba881)
![2](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/d4fae95b-d21d-42d1-9a0f-79b455178c17)
![3](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/dc3ed446-2ea6-4680-ae27-606c0c7ee48a)
![4](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/6ab70984-0176-4bdb-80b9-5a36280e1666)
![5](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/5da24eee-7bdb-46e7-b5cc-52bcd13d2482)
![6](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/c782d952-0601-4059-8e06-ec9d304c1d0b)
![7](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/80082af8-2bcb-4d58-a9ec-339f4af80c96)
![8](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/3efddefc-8715-47c0-b4a6-233078c89746)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
