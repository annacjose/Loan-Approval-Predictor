import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from os import path
import pickle
import flask
from flask import Flask, jsonify, request

# CLEAN FUNCTIONS

def clean_and_drop_dataframe(X):
    x = X.copy()
    x['Gender'].fillna('Male', inplace=True)
    x['Married'].fillna('No', inplace=True)
    x['Self_Employed'].fillna('No', inplace=True)
    x['LoanAmount'].fillna(0, inplace=True)
    x['Loan_Amount_Term'].fillna(0, inplace=True)
    x['Credit_History'].fillna(0, inplace=True)
    x.drop(columns='Loan_ID', inplace=True)
    return x

def convert_dependents_to_int(X):
    x = X.copy()
    x['Dependents'] = x['Dependents'].str.replace('+', '')
    x['Dependents'] = x['Dependents'].str.replace('None', '')
    x['Dependents'] = x['Dependents'].str.strip()
    x['Dependents'].fillna('0', inplace=True)
    x['Dependents'] = x['Dependents'].astype('str')
    x['Dependents'] = x['Dependents'].astype('int')
    return x

# TRANSFORMERS (MORE THAN MEETS THE EYE!)

class GenderTransformer():
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        if path.exists('gender.pkl'):
            self.ohe = pickle.load(open('gender.pkl', 'rb'))
    def fit(self, X, y=None):
        self.ohe.fit(X[['Gender']])
        pickle.dump(self.ohe, open('gender.pkl', 'wb'))
    def transform(self, X, y=None):
        X_ = X.copy()
        ohe_array = self.ohe.transform(X_[['Gender']]).toarray()
        X_['gender1'] = ohe_array[:,0]
        X_['gender2'] = ohe_array[:,1]
        X_.drop(columns='Gender', inplace=True)
        return X_
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

class MarriedTransformer():
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        if path.exists('married.pkl'):
            self.ohe = pickle.load(open('married.pkl', 'rb'))
    def fit(self, X, y=None):
        self.ohe.fit(X[['Married']])
        pickle.dump(self.ohe, open('married.pkl', 'wb'))
    def transform(self, X, y=None):
        X_ = X.copy()
        ohe_array = self.ohe.transform(X_[['Married']]).toarray()
        X_['marital_status1'] = ohe_array[:, 0]
        X_['marital_status2'] = ohe_array[:, 1]
        X_.drop(columns='Married', inplace=True)
        return X_
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

class EducationTransformer():
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        if path.exists('education.pkl'):
            self.ohe = pickle.load(open('education.pkl', 'rb'))
    def fit(self, X, y=None):
        self.ohe.fit(X[['Education']])
        pickle.dump(self.ohe, open('education.pkl', 'wb'))
    def transform(self, X, y=None):
        X_ = X.copy()
        ohe_array = self.ohe.transform(X_[['Education']]).toarray()
        X_['education1'] = ohe_array[:, 0]
        X_['education2'] = ohe_array[:, 1]
        X_.drop(columns='Education', inplace=True)
        return X_
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

class EmploymentTransformer():
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        if path.exists('employed.pkl'):
            self.ohe = pickle.load(open('employed.pkl', 'rb'))
    def fit(self, X, y=None):
        self.ohe.fit(X[['Self_Employed']])
        pickle.dump(self.ohe, open('employed.pkl', 'wb'))
    def transform(self, X, y=None):
        X_ = X.copy()
        ohe_array = self.ohe.transform(X_[['Self_Employed']]).toarray()
        X_['employed1'] = ohe_array[:, 0]
        X_['employed2'] = ohe_array[:, 1]
        X_.drop(columns='Self_Employed', inplace=True)
        return X_
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

class CreditTransformer():
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        if path.exists('credit.pkl'):
            self.ohe = pickle.load(open('credit.pkl', 'rb'))
    def fit(self, X, y=None):
        self.ohe.fit(X[['Credit_History']])
        pickle.dump(self.ohe, open('credit.pkl', 'wb'))
    def transform(self, X, y=None):
        X_ = X.copy()
        ohe_array = self.ohe.transform(X_[['Credit_History']]).toarray()
        X_['credit1'] = ohe_array[:, 0]
        X_['credit2'] = ohe_array[:, 1]
        X_.drop(columns='Credit_History', inplace=True)
        return X_
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

class PropertyTransformer():
    def __init__(self):
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        if path.exists('property.pkl'):
            self.ohe = pickle.load(open('property.pkl', 'rb'))
    def fit(self, X, y=None):
        self.ohe.fit(X[['Property_Area']])
        pickle.dump(self.ohe, open('property.pkl', 'wb'))
    def transform(self, X, y=None):
        X_ = X.copy()
        ohe_array = self.ohe.transform(X_[['Property_Area']]).toarray()
        X_['property1'] = ohe_array[:, 0]
        X_['property2'] = ohe_array[:, 1]
        X_['property3'] = ohe_array[:, 2]
        X_.drop(columns='Property_Area', inplace=True)
        return X_
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_ = self.transform(X, y)
        return X_

# MODEL
rforest = pickle.load(open('rforest.pkl', 'rb'))

# CREATE PIPELINE
clean_and_drop = FunctionTransformer(clean_and_drop_dataframe)
convert_dependents = FunctionTransformer(convert_dependents_to_int)

pipe = Pipeline([
    ('clean_drop', clean_and_drop),
    ('dependents', convert_dependents),
    ('gender', GenderTransformer()),
    ('married', MarriedTransformer()),
    ('education', EducationTransformer()),
    ('employment', EmploymentTransformer()),
    ('credit', CreditTransformer()),
    ('property', PropertyTransformer()),
    ('classifier', rforest)
])

# FLASK

app = Flask('AnnaJose')

@app.route('/hello', methods=['GET'])
def get():
#    return {'message': 'GET not allowed on this API'}
    return 'hello world'

@app.route('/predict', methods=['GET'])
def predict():
    json_data = {'Loan_ID': LP001002, 'Gender': Male, 'Married': No, 'Dependents': 0, 'Education': Graduate, 'Self_Employed': No, 'Applicant_Income': 5849, 'CoApplicantIncome': 0.0, 'LoanAmount': NaN, 'Loan_Amount_Term': 360.0, 'Credit_History': 1.0, 'Property_Area': Urban}

    loan_prediction=rforest.predict(json_data)
    return loan_prediction
    
#     cols = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#        'Loan_Amount_Term', 'Credit_History', 'Property_Area']


#     data = [[]]
#     for i in json_data.values():
#         data[0].append(i)

#     X = pd.DataFrame(data, columns=cols)
#     y_pred = pipe.predict(X)

#     return y_pred[0]


if __name__ == '__main__':
    app.run(host = "0.0.0.0")