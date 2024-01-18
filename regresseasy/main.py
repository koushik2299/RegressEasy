import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

def reg_modelling(X_train, y_train, X_test, y_test):
    models = [
        LinearRegression(), 
        Lasso(), 
        Ridge(),
        RandomForestRegressor(), 
        DecisionTreeRegressor(), 
        xgb.XGBRegressor()
    ]

    results = {}

    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_results = {
            "R2 Score": r2_score(y_test, y_pred),
            "Train Score": model.score(X_train, y_train),
            "Test Score": model.score(X_test, y_test),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "CV Mean Score": np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='r2')),
            "CV Std Deviation": np.std(cross_val_score(model, X_train, y_train, cv=5, scoring='r2'))
        }

        results[model_name] = model_results

        # Print results for each model
        print(f"Results for {model_name}:")
        for key, value in model_results.items():
            print(f"  {key}: {value:.4f}")
        print("-" * 40)

    # Plotting R2 scores with specified modifications
    r2_scores = {model: result["R2 Score"] for model, result in results.items()}
    sorted_models = dict(sorted(r2_scores.items(), key=lambda item: item[1], reverse=True))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_models.keys(), sorted_models.values(), color=sns.color_palette("viridis", len(sorted_models)))
    plt.xlabel('Models')
    plt.ylabel('R2 Score')
    plt.title('R2 Scores of Models in Descending Order')
    plt.xticks(rotation=45)
    plt.ylim([0.5, 1])  # Setting the Y-axis scale to start from 0.5

    # Adding annotations
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

    plt.show()

    return results

# Example usage of the function
# Ensure that you have defined X_train, y_train, X_test, y_test before calling this function
# model_results = reg_modelling(X_train, y_train, X_test, y_test)

# # Access specific model results, e.g., Linear Regression
# print("Linear Regression Results:", model_results["LinearRegression"])
