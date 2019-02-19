import pandas as pd
from sklearn.cross_validation import train_test_split # train/test split.
from sklearn.linear_model import LinearRegression # Modeling.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import statsmodels.formula.api as sm

data = pd.read_csv('tennis.csv')

#LE 
outlook = data.iloc[:,0:1].values
l_encoder_outlook = LabelEncoder()
outlook[:,0] = l_encoder_outlook.fit_transform(outlook[:,0])

#OHE 
ohe = OneHotEncoder(categorical_features='all')
outlook=ohe.fit_transform(outlook).toarray()

#LE
true_false = data.iloc[:,3:4].values
l_encoder_truefalse = LabelEncoder()
true_false[:,0] = l_encoder_truefalse.fit_transform(true_false[:,0])

#OHE 
ohe = OneHotEncoder(categorical_features='all')
true_false=ohe.fit_transform(true_false).toarray()


#LE 
yes_no = data.iloc[:,4:5].values
l_encoder_yes_no = LabelEncoder()
yes_no[:,0] = l_encoder_yes_no.fit_transform(yes_no[:,0])

#OHE
ohe = OneHotEncoder(categorical_features='all')
yes_no=ohe.fit_transform(yes_no).toarray()

# With OneHotEncoder and LabelEncoder, we made the string data to numerical data. This is an important part of data preprocessing.
# "True or false", "yes or no" not understandable for computer. So we made them numerical.

temp = data.iloc[:,1:2].values #Taking temperature column.
humidity = data.iloc[:,2:3].values #Taking humidity column.

## Creating columns with their names.
outlookDF = pd.DataFrame(data = outlook, columns = ['overcast', 'sunny', 'rainy'])
tempDF = pd.DataFrame(data = temp, columns = ['temperature'])
humidityDF = pd.DataFrame(data = humidity, columns = ['humidity'])
windyDF = pd.DataFrame(data = true_false, columns = ['true', 'false'])
playDF = pd.DataFrame(data = yes_no, columns = ['yes', 'no'])

windyDF = windyDF.iloc[:,-1] # only 1 column taking (protecting from dummy variable.)
playDF = playDF.iloc[:,-1] # only 1 column taking (protecting from dummy variable.)

completed_data = pd.concat([outlookDF, tempDF, humidityDF], axis = 1) # completed data.

x_train, x_test,y_train,y_test = train_test_split(completed_data,playDF,test_size=0.33, random_state=0) # split test/train data. 
## Data preprocessing has done.

# Regression model.
regressor = LinearRegression()
regressor.fit(x_train,y_train)
prediction = regressor.predict(x_test)

#stats model p-value - backward elimination
X = np.append(arr = np.ones((14,1)).astype(int), values = completed_data, axis = 1)
X_list = completed_data.iloc[:,[0,1,2]].values # 3rd column eliminated from completed_data because p>|t| value was too low. Prediction is more successful now.
regression_ols = sm.OLS(endog= playDF, exog=X_list).fit()

print(regression_ols.summary())

## Original data --->> y_test Predict data ---->>>> prediction
## You can compare them. 

"""
print("=== ORIGINAL DATA ===") 
print(y_test)  #First column is holding index values. Second column is holding original data. (1 = true)
print("=== PREDICT DATA ===")
print(prediction)
print("\n\n\n")

# You can add variables in completed_data. For example you can add windyDF. At 55.line you can see p-values and prediction values again. 
# Predictions will be change.                                           
"""
