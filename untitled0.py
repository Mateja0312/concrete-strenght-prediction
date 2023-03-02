import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def prepData(csvFile):

  df = pd.read_csv('Concrete_Data.csv')
  columnRenames = {
      'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
      'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Slag',
      'Fly Ash (component 3)(kg in a m^3 mixture)': 'Ash',
      'Water  (component 4)(kg in a m^3 mixture)': 'Water',
      'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Plasticizer',
      'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'CoarseAgg',
      'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'FineAgg',
      'Age (day)': 'Age',
      'Concrete compressive strength(MPa, megapascals) ': 'Strength',
  }
  df.rename(columns=columnRenames, inplace=True)
  
  df = df.sample(frac=1) #row shuffler (for cross validation)

  x = df.drop(["Strength", 'CoarseAgg', 'FineAgg', "Plasticizer"], axis = 1)
  y = df["Strength"]
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state = random.randint(0,100))

  return x, y, x_train, x_test, y_train, y_test

def model_evaluation(y, y_predicted):
  
  mse = mean_squared_error(y, y_predicted) 
  mae = mean_absolute_error(y, y_predicted) 
  rmse = np.sqrt(mse)
  r2 = r2_score(y, y_predicted)
  
  res = pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
  res.columns = ['y', 'y_pred']
  
  print(res.head(10))
  print('Mean squared error: ', mse)
  print('Mean absolute error: ', mae)
  print('Root mean squared error: ', rmse)
  print('R2 score: ', r2)
  print('Spearman correlation', spearmanr(y, y_predicted).correlation)
  
  return r2

def cross_evaluation(model, x, y):
    crossValRuns = 10
    scores = cross_val_score(model, x, y, cv=crossValRuns, scoring="r2")
    print(f"Cross validation ({crossValRuns} iterations)")
    print(f"\t score (r2) = {scores.mean()}")
    print(f"\t standard deviation = {scores.std()}")
    #print("scores = ", scores)

def testModel(model, x, y, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    
    y_predicted = model.predict(x_test)
    model_evaluation(y_test, y_predicted)
    
    modelName = type(model).__name__
    fig = plt.figure(figsize=[6,5])  
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.suptitle(modelName, fontsize=25)
    plt.scatter(y_test,y_predicted,  s=100)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    plt.ylabel('Strength - Predicted [MPa]', fontsize=16)
    plt.xlabel('Strenght - Actual [MPa]', fontsize=16)
    plt.savefig(modelName+'.png')
    plt.show()
    
    cross_evaluation(model, x, y)
    
data_sets  = prepData('Concrete_Data.csv')

print(" \n LINEAR MODEL")
linRM = LinearRegression(fit_intercept = True)
testModel(linRM, *data_sets)

print(" \n SUPPORT VECTOR - RADIAL BASIS KERNEL")
SVR_RBF =  SVR(kernel="rbf", C=100, gamma=0.0001, epsilon=0.1)
testModel(SVR_RBF, *data_sets)

print(" \n RANDOM FOREST")
randomForest = RandomForestRegressor(n_estimators=10, max_features=2)
testModel(randomForest, *data_sets)






