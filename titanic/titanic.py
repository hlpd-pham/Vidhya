# Titanic datset

# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing datasets
#gender = pd.read_csv('gender_submission.csv')

# Filter out irrelevant columns
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
              axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],
              axis=1, inplace=True)

# Get most frequent item any column
#def most_common(lst):
#    return max(set(lst), key=lst.count)
#train_vals = train_df.values
#last_col = []
#for item in train_vals[:, -1]:
#    last_col.append(item)
#print(most_common(last_col))

# Taking care of missing data (Fill NaN values)
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
train_df = DataFrameImputer().fit_transform(train_df)
y_train = train_df.iloc[:, 0].values
train_df.drop(['Survived'], axis=1, inplace=True)
X_train = train_df.values

# Encoding categorical data for train set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_train = LabelEncoder()
X_train[:, 1] = labelencoder_train.fit_transform(X_train[:, 1])
X_train[:, -1] = labelencoder_train.fit_transform(X_train[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_train = onehotencoder.fit_transform(X_train).toarray()
# Avoid Dummy Variable Trap
X_train = X_train[:, 1:]

# Encoding categorical data for test set
test_df = DataFrameImputer().fit_transform(test_df)
X_test = test_df.values
labelencoder_test = LabelEncoder()
X_test[:, 1] = labelencoder_test.fit_transform(X_test[:, 1])
X_test[:, -1] = labelencoder_test.fit_transform(X_test[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [6])
X_test = onehotencoder.fit_transform(X_test).toarray()
# Avoid Dummy Variable Trap
X_test = X_test[:, 1:]

# Importing test set result
y_test_df = pd.read_csv('gender_submission.csv')
y_test = y_test_df.iloc[:, 1].values

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)

# Fitting SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(X_train, y_train)

# Fitting Logistic Regression to the Training set - best?
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#result_set = pd.read_csv('result.csv')
#y_pred = result_set.iloc[:, 1].values


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
