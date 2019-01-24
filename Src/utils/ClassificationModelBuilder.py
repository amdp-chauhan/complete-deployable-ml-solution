# Library Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import dill as pickle
from .DataPreparation import *


# Ignoring all warnings
import warnings
warnings.filterwarnings("ignore")
import os

# read dataset
train_df = pd.read_csv('./Data/train.csv')

# global random state
rand_state_ = 42

"""
    After analysing, visualising and going through few kernels, my observations are:
        - Features like 'PassengerId' and 'Ticket' seems of no use, so we should remove them.
        - 'Age' and 'Fare' are of type float, so it will be better if we take their ceil value and convert them to integer.
        - 'Cabin' contains more than 77% of missing values, so it should get discarded because we are not expected to fill this many missing values on our own and even if we do then model will loose its significance.
        - From 'Name', we can create a new feature called 'Title' It may help model in performing better, not sure though, will see.
        - We have 'SibSp' and 'Parch' features which shows person's Siblings, Spouse, Parents and children. We can use this feature to come with two new features called 'FamilySize' and 'IsAlone'.
        - Newly created feature 'FamilySize' can be used to create a new feature called 'FarePerHead'.
""" 


"""
    This class will take care of model's hyper parameter tuning, prediction on train/test dataset and evaluation
"""
class Modeling(object):

    def __init__(self, test_train_ratio):

        self.classifiers = {}
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(train_df.drop(['Survived'], axis=1), train_df.Survived, test_size=test_train_ratio, random_state=rand_state_)


    def evaluate_model(self, modelName, train_predictions, test_predictions):

        self.classifiers[modelName] = {
            'TrainingAccuracy': accuracy_score(self.y_train, train_predictions),            
            'TestAccuracy': accuracy_score(self.y_test, test_predictions)
        }


    def fit_and_predict_using_RandomSearchCV(self, classifier):
        random_cv_model = RandomizedSearchCV(estimator=classifier['instance'], param_distributions=classifier['param_grid'], cv=10)
        random_cv_model.fit(self.X_train, self.y_train)
        self.evaluate_model(classifier['name'], random_cv_model.predict(self.X_train), random_cv_model.predict(self.X_test))
        self.classifiers[classifier['name']]['Estimator'] = random_cv_model.estimator

        return self.classifiers[classifier['name']]


    def voting_classifier(self, classifier_names):

        selected_classifiers = [(classifier_name, self.classifiers[classifier_name]['Estimator']) for classifier_name in classifier_names]
        voting_classifier = VotingClassifier(estimators=selected_classifiers, voting='hard')
        voting_classifier.fit(self.X_train, self.y_train)
        self.evaluate_model(voting_classifier.__class__.__name__, voting_classifier.predict(self.X_train), voting_classifier.predict(self.X_test))
        self.evaluate_model(voting_classifier.__class__.__name__, voting_classifier.predict(self.X_train), voting_classifier.predict(self.X_test))
        self.classifiers[voting_classifier.__class__.__name__]['Estimator'] = voting_classifier
        return self.classifiers[voting_classifier.__class__.__name__]


"""
    Processing Datasets
"""
data_preparation = DataPreparation()
train_df = data_preparation.preprocess(train_df)


"""
    Building and comparing Models
"""
model_ops = Modeling(3/10)

classifiers = [
    {
        'name': 'DecisionTreeClassifier',
        'instance': DecisionTreeClassifier(),
        'param_grid': {
            'splitter': ['best', 'random'],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4],
            'min_samples_split': [2, 3, 4],
            'max_features': ['sqrt'],
            'random_state': [rand_state_]
        }
    }, {
        'name': 'RandomForestClassifier',
        'instance': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [10, 30, 60, 90, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4],
            'min_samples_split': [2, 3, 4],
            'max_features': ['sqrt'],
            'random_state': [rand_state_]
        }
    }, {
        'name': 'XGBClassifier',
        'instance': XGBClassifier(),
        'param_grid': {
            'max_depth': [3, 4, 5],
            'learning_rate': [.1, .06, .03, .01],
            'n_estimators': [80, 100, 120],
            'booster': ['gbtree', 'gblinear', 'dart'],
            'gamma': [0, 2, 4],
            'random_state': [rand_state_]
        }
    }, {
        'name': 'KNeighborsClassifier',
        'instance': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [5, 6, 7, 8, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
    }, {
        'name': 'ExtraTreesClassifier',
        'instance': ExtraTreesClassifier(),
        'param_grid': {
            'n_estimators': [20, 40, 80],
            'min_samples_split': [2, 3, 4],
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt']
        }
    }, {
        'name': 'RidgeClassifierCV',
        'instance': RidgeClassifierCV(),
        'param_grid': {
            'alphas': [(0.05, 0.1, 0.5, 1, 2)]
        }
    }, {
        'name': 'AdaBoostClassifier',
        'instance': AdaBoostClassifier(),
        'param_grid': {
            'base_estimator': [ 
                # Decided hyper parameter values after RandomSearch Cross Validation
                DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=1, min_samples_split=3, random_state=rand_state_, splitter='best'),
                XGBClassifier(booster='dart', gamma=2, learning_rate=0.1, max_depth=3, n_estimators=100, random_state=rand_state_)
            ],
            'n_estimators': [50, 70, 90],
            'random_state': [rand_state_],
            'algorithm': ['SAMME', 'SAMME.R'],
            'learning_rate': [0.8, 1.0, 1.3]
        }
    }, {
        'name': 'BaggingClassifier',
        'instance': BaggingClassifier(),
        'param_grid': {
            'base_estimator': [
                # Decided hyper parameter values after RandomSearch Cross Validation
                DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=1, min_samples_split=3, random_state=rand_state_, splitter='best'),
                XGBClassifier(booster='dart', gamma=2, learning_rate=0.1, max_depth=3, n_estimators=100, random_state=rand_state_)
            ],
            'n_estimators': [10, 20, 30],
            'random_state': [rand_state_],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
        }
    }, {
        'name': 'GradientBoostingClassifier',
        'instance': GradientBoostingClassifier(),
        'param_grid': {
            'loss': ['deviance', 'exponential'],
            'n_estimators': [100, 120, 150],
            'random_state': [rand_state_],
            'min_samples_split': [2, 3, 4],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 3, 4]
        }
    }
]


for classifier in classifiers:
    classifier_performance = model_ops.fit_and_predict_using_RandomSearchCV(classifier)
    print(f"{classifier['name']} Performance - \n{classifier_performance}")


# We are going to use the SimpleVoting Ensemble technique to make our final predictions.
voting_classifier = model_ops.voting_classifier(['AdaBoostClassifier', 'BaggingClassifier'])
print(f'VotingClassifier Performance - \n{voting_classifier}')


# Comparing performance of Classifiers 
score_df = pd.DataFrame([{'ModelName': name, 'Test Accuracy': props['TestAccuracy'], 'Training Accuracy': props['TrainingAccuracy']} for name, props in model_ops.classifiers.items()])
score_df.set_index('ModelName')
print(f'All Classifiers Performance Table - \n{score_df}')


# We are saving VotingClassifier trained instance, if you want you can save any other model/s as well.
filename = 'voting_classifier_v1.pk'
with open('./Src/ml-model/'+filename, 'wb') as file:
    pickle.dump(voting_classifier['Estimator'], file)