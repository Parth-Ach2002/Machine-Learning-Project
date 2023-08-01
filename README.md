# Machine-Learning-Project
#Creating a machine learning algorithm to use regression analysis to predict acceleration of a car based on curb weight and horse power. Also visualising the predictions against the actual values in a line graph. This project was done using Python and various libraries within.
import pandas as pd 
data=pd.read_excel('ML final Dataset.xlsx')
x=data[["Horsepower","Curb Weight (lbs)"]]
y=data['Acceleration']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
a=int(input("Enter Horsepower:"))
b=int(input("Enter Weight:"))
c=pd.DataFrame({"Horsepower":[a],"Weight":[b]})
y_pred=lr.predict(c)
y_pred
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
y_pred2=dtr.predict(c)
y_pred2
ax = range(len(y_test))
plt.plot(ax, y_test, linewidth=1.7, label="Original",color="black")
plt.plot(ax, y_pred2, linewidth=0.8, label="Decision Tree Regressor")
plt.plot(ax,y_pred,lw=0.8, label="Multiple Linear Regression")
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 

