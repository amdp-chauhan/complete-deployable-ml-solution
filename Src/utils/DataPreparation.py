# Library Import
import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.preprocessing import LabelEncoder
import re


"""
    This class will take care of data preprocessing and feature engineering.
"""
class DataPreparation():

    def __init__(self):
        pass


    def preprocess(self, df):
        df = self.fill_missing_values(df)
        df = self.feature_extraction(df)
        df = self.handle_categorical_variables(df)
        df = self.dimensionality_reduction(df)

        return df


    def fill_missing_values(self, df):
        df.Age = np.ceil(df.Age.fillna(df.Age.median())).astype(int)
        df.Embarked = df.Embarked.fillna(df.Embarked.mode()[0])
        df.Fare = np.ceil(df.Fare.fillna(df.Fare.mean())).astype(int)
        
        return df


    def feature_extraction(self, df):
        # FamilySize includes a person, his parents, siblings, children and spouse 
        df['FamilySize'] = df.SibSp + df.Parch + 1
        df['FarePerHead'] = (df.Fare/df.FamilySize).astype(int)
        df['IsAlone'] = df.FamilySize.apply(lambda x: 1 if x==1 else 0)
        # In AgeGroup, we have divided population into four groups - kid, teen, adult & old
        df['AgeGroup'] = df.Age.apply(lambda x: 'kid' if x<13 else 'teen' if x<20 else 'adult' if x<41 else 'old')

        # adding Title variable
        df['Title'] = df.Name.apply(lambda x: re.search('(?<=, )\w+', x).group(0))
        df.Title.replace(to_replace=['Ms', 'Lady', 'the', 'Dona'], value='Mrs', inplace=True)
        df.Title.replace(to_replace=['Mme', 'Mlle'], value='Miss', inplace=True)
        df.Title.replace(to_replace=['Jonkheer', 'Sir', 'Capt', 'Don', 'Col', 'Major', 'Rev', 'Dr'], value='Mr', inplace=True)

        return df


    def handle_categorical_variables(self, df):
        df = pd.get_dummies(df, drop_first=True, columns=['Sex', 'Embarked'])
        df.AgeGroup = LabelEncoder().fit_transform(df.AgeGroup)
        df.Title = LabelEncoder().fit_transform(df.Title)

        return df


    def dimensionality_reduction(self, df):

        return df.drop(labels=['PassengerId','Name','Ticket', 'Cabin'], axis=1)
