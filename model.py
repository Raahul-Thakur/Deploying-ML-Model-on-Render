import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import sklearn

print('numpy', np.__version__)
print('pandas', pd.__version__)
print('sklearn', sklearn.__version__)

df = pd.read_csv('placement-dataset.csv')

df.head()

df = df.drop('Unnamed', axis = 1)

X = df.drop('placement',axis=1)
y = df['placement']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100,200,300],
    'max_depth': [None, 10,20,30],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train,y_train)

best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
print(classification_report(y_test, y_pred))

with open('model.pkl','wb') as file:
    pickle.dump(best_rf,file)
