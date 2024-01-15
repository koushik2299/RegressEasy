# RegressModel

RegressModel is a Python package designed for streamlined regression analysis. It facilitates the process of training and evaluating multiple regression models on a dataset, providing a comprehensive comparison of their performance. This package is particularly useful for data scientists and machine learning practitioners who need to quickly assess the best regression model for their specific data.

## Features

- Easy comparison of various regression models.
- Includes common models like Linear Regression, Lasso, Ridge, SVR, Random Forest, Decision Tree, and XGBoost.
- Outputs performance metrics like R2 score, RMSE, MSE, MAE, and cross-validation scores.

## Installation

You can install RegressModel using pip:

```bash
pip install regressmodel
```
## Usage

Here is a simple example of how to use RegressModel:

```bash
from regressmodel.core import RegModelling
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset
df = pd.read_csv('your_dataset.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Using RegressModel
results = RegModelling(X_train, y_train, X_test, y_test)

# The results variable will contain performance metrics of the models
```
## Requirements
Python 3.6 or higher
scikit-learn
numpy
xgboost
## Contributing
Contributions to RegressModel are welcome! Feel free to fork the repository and submit pull requests.

## License
RegressModel is licensed under the MIT License - see the LICENSE file for details.

## Author
**[SaiKoushikGandikota](https://www.linkedin.com/in/gandikota-sai-koushik/)**
