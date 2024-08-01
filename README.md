# Linear Regression Project - Solutions

## Overview
This project involves analyzing customer data from a New York City clothing store that sells products both online and in person. Customers often receive in-store advice and later make purchases through the mobile app or website. The company aims to determine whether to focus on improving the mobile app experience or the website.

## Data Description
The dataset contains the following columns:
- **Avg. Session Length:** Average duration of in-store style advice sessions.
- **Time on App:** Average time spent on the app in minutes.
- **Time on Website:** Average time spent on the website in minutes.
- **Length of Membership:** Number of years the customer has been a member.
- **Yearly Amount Spent:** Total dollars spent by the customer annually.

## Data Import and Initial Inspection
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

customers = pd.read_csv("Ecommerce Customers.csv")

customers.head()
customers.info()
customers.describe()
```

## Exploratory Data Analysis
We will focus on the numerical data to understand the relationships between different variables.

### Jointplots
```python
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
```

### Pairplot
```python
sns.pairplot(customers)
```

### Linear Model Plot
```python
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)
```

## Training and Testing Data
We split the data into training and testing sets to evaluate our model.

### Split Data
```python
from sklearn.model_selection import train_test_split

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

## Training the Model
We use a Linear Regression model to fit our training data.

### Fit Model
```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)
print('Coefficients: \n', lm.coef_)
```

## Predicting Test Data
We evaluate the model's performance using the test data.

### Predictions
```python
predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```

## Model Evaluation
We calculate various metrics to evaluate the model's performance.

### Metrics
```python
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

sns.distplot((y_test - predictions), bins=50)
```

## Conclusion
To decide whether to focus on the mobile app or the website, we interpret the coefficients of the model.

### Coefficients Interpretation
```python
coefficients = pd.DataFrame(lm.coef_, X.columns)
coefficients.columns = ['Coefficient']
coefficients
```

### Insights
- **Avg. Session Length:** A 1 unit increase is associated with a $25.98 increase in yearly spending.
- **Time on App:** A 1 unit increase is associated with a $38.59 increase in yearly spending.
- **Time on Website:** A 1 unit increase is associated with a $0.19 increase in yearly spending.
- **Length of Membership:** A 1 unit increase is associated with a $61.27 increase in yearly spending.

### Recommendation
While both the app and website contribute to yearly spending, the app shows a stronger correlation. However, further analysis of the relationship between Length of Membership and the app/website could provide additional insights.

### Reference
For detailed code and further analysis, refer to the [project notebook](https://github.com/monahatami1/monogram1/blob/master/notebook2-week8-Linear%20Regression%20Project%20-%20Solutions.ipynb).
