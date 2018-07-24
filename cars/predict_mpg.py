# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import scatter_matrix

#in_file = open('car.txt', 'r')
#out_file = open('car.csv','w')
#
##out_file.write('mpg,cylinders,displacement,'+
##               'horsepower,weight,acceleration,'+
##               'model year, origin,car name\n')
#for line in in_file:
#    split = line.split()
#    for item in split:
#        out_file.write(item +',')
#    out_file.write('\n')
#
#out_file.close()
#in_file.close()

# Importing the dataset
dataset = pd.read_csv('car.csv',header = None)
X = dataset.iloc[:, 1:8].values
y = dataset.iloc[:, 0].values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
#X[:, -1] = labelencoder.fit_transform(X[:, -1])
#onehotencoder = OneHotEncoder(categorical_features = [306])
#X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:3])
X[:, 0:3] = imputer.transform(X[:, 0:3])

# Create test sets and training sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#sum = 0
## calculate accuracy
#for i in range(len(y_pred)):
#    displacement = abs(float(y_pred[i] / y_test[i]) - float(y_test[i]))
#    print(displacement)
#    sum += displacement
#accuracy = sum / len(y_pred)
#print("The average displacement of the model is " + str(accuracy))
    
