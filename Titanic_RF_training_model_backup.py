#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#data = pd.read_csv('titanic/train.csv')
data = pd.read_csv('/Users/hans/Desktop/Main/WebDev/python/data science/titanic/train.csv')
#df_test = pd.read_csv('test.csv')

#data cleaning
#
#
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'No title in name'

#map - apply the function to each one of the value in the list
titles = set([x for x in data.Name.map(lambda x: get_title(x))])

def shorter_titles(x):
    title=x['Title']
    if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
    elif title in ['Jonkheer', 'Don', 'the Countess', 'Dona','Lady', 'Sir']:
        return 'Royalty'
    elif title in ['Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    else:
        return title

#Creating - feature engineering
#
#
#data['Title']=data['Name'].map(lambda x:get_title(x))
data['Title']=list(map(lambda x:get_title(x), data['Name']))
data['Title']=data.apply(shorter_titles, axis=1)
#use median for age
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
#replace missing value with the most common one
data['Embarked'].fillna('S', inplace=True)

#drop data
del data['Cabin']
data.drop('Name',axis=1, inplace=True)
data.drop('Ticket',axis=1, inplace=True)
data.Sex.replace(('male','female'),(0,1), inplace=True)
data.Embarked.replace(('S','C','Q'),(0,1,2), inplace=True)
data.Title.replace(('Mr', 'Mrs', 'Miss', 'Master' ,'Royalty', 'Rev' ,'Dr', 'Officer'),(0,1,2,3,4,5,6,7), inplace=True)

#set x and y
y = data['Survived']
x = data.drop(['Survived', 'PassengerId'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.1) 

#the model
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
#have a test run to see accuracy
y_pred = randomforest.predict(x_test)

#store RF prediction model
filename='titanic_model.sav'
pickle.dump(randomforest, open(filename, 'wb'))


# In[ ]:




