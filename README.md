## API-First approach to make Machine Learning solution usable

### Introduction
In this application I have solved 'Titanic Survival Prediction problem' and using simple **VotingClassifier** to make predictions. I have trained many other models as well and if you want to then you can view their performance and can choose any of the desired model (by making some changes in `Src/utils/ClassificationModelBuilder.py` file) to make predictions. 

### Application Setup
I have used Python's  `venv` module for creating/managing virtual environment and `flask` framework for API creation. 

If you're not much aware of `venv` environment setup, than you can go through [this](https://docs.python.org/3/tutorial/venv.html) documentation. I learnt from same.

Once you have `venv` installed and got basic understanding, follow below steps to run this application:
1. `git clone https://github.com/amdp-chauhan/titanic-survival-complete-ml-solution.git` && `cd titanic-survival-complete-ml-solution`
2. `python -m venv ./` - It will create a virtual environment in application directory.
3. `Scripts\activate.bat` - It will run this virtual environment.
4. `pip install -r packages.txt` - it will install all required dependencies.
5. `python application.py` - it will run the application.
6. `deactivate` - If you want to exit from virtual environment.

Upper commands will work fine in Windows 10, for Linux you can find alternatives in venv documentation.

> Note that in `application.py` file, second import statement is commented out, it is because if it is enabled then it starts retraining classifier models, which is not required if you already have created a final model and data-set is same. Final models exists in `Src/ml-model/voting_classifier_v1.pk` we use same model to make predictions for requested JSON record.

### Making Predictions 
For predictions I have created an POST API:
 ```
http://{domain}/titanic-survival-classification-model/predict
 ```

It accepts list of JSON of test records and in return will give you a predicted Survival values in 0/1.

For example, for below input parameters: 

```
[{
    "PassengerId": 892,
    "Pclass": 3,
    "Name": "Kelly, Mr. James",
    "Sex": "male",
    "Age": 34.5,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 330911,
    "Fare": 7.8292,
    "Cabin": "",
    "Embarked": "Q"
  },{
    "PassengerId": 893,
    "Pclass": 3,
    "Name": "Wilkes, Mrs. James (Ellen Needs)",
    "Sex": "female",
    "Age": 47,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": 363272,
    "Fare": 7,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 894,
    "Pclass": 2,
    "Name": "Myles, Mr. Thomas Francis",
    "Sex": "male",
    "Age": 62,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 240276,
    "Fare": 9.6875,
    "Cabin": "",
    "Embarked": "Q"
  },{
    "PassengerId": 895,
    "Pclass": 3,
    "Name": "Wirz, Mr. Albert",
    "Sex": "male",
    "Age": 27,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 315154,
    "Fare": 8.6625,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 896,
    "Pclass": 3,
    "Name": "Hirvonen, Mrs. Alexander (Helga E Lindqvist)",
    "Sex": "female",
    "Age": 22,
    "SibSp": 1,
    "Parch": 1,
    "Ticket": 3101298,
    "Fare": 12.2875,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 897,
    "Pclass": 3,
    "Name": "Svensson, Mr. Johan Cervin",
    "Sex": "male",
    "Age": 14,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 7538,
    "Fare": 9.225,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 898,
    "Pclass": 3,
    "Name": "Connolly, Miss. Kate",
    "Sex": "female",
    "Age": 30,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 330972,
    "Fare": 7.6292,
    "Cabin": "",
    "Embarked": "Q"
  },{
    "PassengerId": 899,
    "Pclass": 2,
    "Name": "Caldwell, Mr. Albert Francis",
    "Sex": "male",
    "Age": 26,
    "SibSp": 1,
    "Parch": 1,
    "Ticket": 248738,
    "Fare": 29,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 900,
    "Pclass": 3,
    "Name": "Abrahim, Mrs. Joseph (Sophie Halaut Easu)",
    "Sex": "female",
    "Age": 18,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 2657,
    "Fare": 7.2292,
    "Cabin": "",
    "Embarked": "C"
  },{
    "PassengerId": 901,
    "Pclass": 3,
    "Name": "Davies, Mr. John Samuel",
    "Sex": "male",
    "Age": 21,
    "SibSp": 2,
    "Parch": 0,
    "Ticket": "A/4 48871",
    "Fare": 24.15,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 902,
    "Pclass": 3,
    "Name": "Ilieff, Mr. Ylio",
    "Sex": "male",
    "Age": "",
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 349220,
    "Fare": 7.8958,
    "Cabin": "",
    "Embarked": "S"
  },{
    "PassengerId": 903,
    "Pclass": 1,
    "Name": "Jones, Mr. Charles Cresson",
    "Sex": "male",
    "Age": 46,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": 694,
    "Fare": 26,
    "Cabin": "",
    "Embarked": "S"
}]
```
We will get below output:
```
{
	"predictions": "[{\"PassengerId\":892,\"Survived\":1},{\"PassengerId\":893,\"Survived\":1},{\"PassengerId\":894,\"Survived\":0},{\"PassengerId\":895,\"Survived\":1},{\"PassengerId\":896,\"Survived\":1},{\"PassengerId\":897,\"Survived\":1},{\"PassengerId\":898,\"Survived\":0},{\"PassengerId\":899,\"Survived\":0},{\"PassengerId\":900,\"Survived\":1},{\"PassengerId\":901,\"Survived\":0},{\"PassengerId\":902,\"Survived\":1},{\"PassengerId\":903,\"Survived\":0}]"
}
```