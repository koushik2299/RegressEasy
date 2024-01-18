from regresseasy import RegModelling
import pandas as pd

df = pd.read_csv("gemstone.csv")

## Independent and dependent features
X = df.drop(labels=['price'],axis=1)
Y = df[['price']]