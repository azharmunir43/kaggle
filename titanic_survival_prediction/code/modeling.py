import warnings
import os
import dill as pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")


def build_and_train():
    data = pd.read_csv('../Data/train_processed.csv')

    feature_vector = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']
    x_train = data[feature_vector]
    y_train = data['Survived']
    pipe = make_pipeline(PreProcessing(),
                         RandomForestClassifier())

    param_grid = {"randomforestclassifier__n_estimators": [10, 20, 30],
                  "randomforestclassifier__max_depth": [None, 3, 5, 6],
                  "randomforestclassifier__max_leaf_nodes": [None, 5, 10, 15],
                  "randomforestclassifier__min_impurity_split": [0.1, 0.2, 0.3]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    grid.fit(x_train, y_train)
    return grid


class PreProcessing(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, df):

        feature_vector = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']
        df['Pclass'] = df['Pclass'].astype(int)
        df['Age'] = df['Age'].astype(int)
        df['SibSp'] = df['SibSp'].astype(int)
        df['Parch'] = df['Parch'].astype(int)
        df['Fare'] = df['Fare'].astype(int)
        df = df[feature_vector]
        sex_values = {'female': 0, 'male': 1}
        embarked_values = {'S': 0, 'C': 1, 'Q': 2}
        df.replace({'Sex': sex_values, 'Embarked': embarked_values}, inplace=True)

        return df.as_matrix()

    def fit(self, df, y=None, **fit_params):
        return self


if __name__ == '__main__':
    print(os.path.dirname(os.path.abspath(__file__)))
    model = build_and_train()

    filename = 'model_v1.pk'
    with open('D:\Workplace\Self\Django\Insightish\media' + filename, 'wb') as file:
        pickle.dump(model, file)
