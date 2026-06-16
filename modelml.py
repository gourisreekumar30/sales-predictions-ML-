import pandas as pd
df=pd.read_csv('Advertising.csv')
X=df[['TV','Radio','Newspaper']]
y=df['Sales']
print("Features(X)")
print(X.head())
print("Target(y)")
print(y.head())
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)
print("testing rows",len(X_test))
print("training rows",len(X_train))
from sklearn.linear_model import LinearRegression
model=LinearRegression()# training empty model
model.fit(X_train,y_train)
print(model.coef_)
print(model.intercept_)
predictions=model.predict(X_test)
results=pd.DataFrame({
    'Actual':y_test.values,
    'Predicted':predictions.round(1)
})
print(results.head(10))
from sklearn.metrics import(
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
mae=mean_absolute_error(y_test,predictions)
mse=mean_squared_error(y_test,predictions)
rmse=np.sqrt(mse)
r2=r2_score(y_test,predictions)
print("\n--MODEL ACCURACY--")
print("MAE",mae)
print("MSE",mse)
print("R2",r2)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.scatter (
    range(len(y_test)),
    y_test,
    color='blue',
    label='Actual sales'
)
plt.scatter(
    range(len(predictions)),
    predictions,
    color='red',
    label='Predicted sales'
)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Data Points')
plt.ylabel('Sales')
plt.legend()
plt.show()
plt.close()
print("MY OWN PREDICTIONS")
my_budget=pd.DataFrame({
    'TV':[300],
    'Radio':[50],
    'Newspaper':[30]
})
new_predictions=model.predict(my_budget)
print("Predicted Sales →", round(new_predictions[0], 1), "units!")