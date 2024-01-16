import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def RegModelling(X_train, y_train, X_test, y_test):
    # Initialize lists for metrics
    R2_score = []
    Score_Train = []
    Score_Test = []
    RMSE = []
    MAE = []
    MSE = []
    mean = []
    std = []

    # List of regression models
    regg_models = [
        LinearRegression(), 
        Lasso(), 
        Ridge(), 
        SVR(),
        RandomForestRegressor(), 
        DecisionTreeRegressor(), 
        xgb.XGBRegressor()
    ]

    # Dictionaries for storing scores
    R2_score_dict = {}
    RMSE_score_dict = {}
    MAE_score_dict = {}
    MSE_score_dict = {}
    Score_Train__dict = {}
    Score_Test__dict = {}
    Cross_Valication_score_dict = {}

    for model in regg_models:    
        # Train model
        train_model = model.fit(X_train, y_train)
        y_pred = train_model.predict(X_test)

        # Compute metrics
        r2score = metrics.r2_score(y_test, y_pred)
        R2_score.append(r2score)

        scoretrain = train_model.score(X_train, y_train)
        Score_Train.append(scoretrain)

        scoretest = train_model.score(X_test, y_test)
        Score_Test.append(scoretest)

        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        RMSE.append(rmse)

        mse = metrics.mean_squared_error(y_test, y_pred)
        MSE.append(mse)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        MAE.append(mae)

        cvs = cross_val_score(model, X_train, y_train, cv=5, scoring=make_scorer(r2_score))
        mean.append(np.mean(cvs))
        std.append(np.std(cvs))

    # Print the scores
    print_scores("r2_score", regg_models, R2_score)
    print_scores("Score Train", regg_models, Score_Train)
    print_scores("Score Test", regg_models, Score_Test)
    print_scores("Normalized RMSE", regg_models, RMSE)
    print_scores("MSE", regg_models, MSE)
    print_scores("MAE", regg_models, MAE)
    print_scores("Cross Validation Score", regg_models, mean)

    def print_scores(metric_name, models, scores):
        print(f"            {metric_name} \n")
        for i, model in enumerate(models):
            print(model.__class__.__name__, ':', scores[i])
        print('-' * 60, '\n')


