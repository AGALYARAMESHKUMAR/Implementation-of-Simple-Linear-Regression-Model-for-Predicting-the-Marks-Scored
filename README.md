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

Developed by: Agalya R
RegisterNumber: 212222040003

# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)


```


## Output:
To read csv file
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/1efb44fd-3ae2-4e63-807f-2431bb28d587)
To Read Head and Tail Files
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/67966a3a-3e3d-435f-b983-a33ed2be58bb)
Compare Dataset
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/c5fdf61a-5e0a-47e3-9086-ebdab78e9082)
Predicted Value
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/aef3d4b3-ff40-4e04-9b99-da1a3717a48d)
Graph For Training Set
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/568c4d83-51ee-454d-b018-c52c80a5dcbb)
Graph For Testing Set
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/09487efd-a26d-401f-bcfd-2cd4bfec5c00)
Error
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394395/4f018176-b0f7-4bdd-b61f-e44208672089)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
