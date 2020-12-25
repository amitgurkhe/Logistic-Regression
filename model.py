import pandas as pd 
import pickle as pkl

A=pd.read_csv("C:/Users/SAMSUNG/deploy/50_Startups - 50_Startups.csv")


X = A[['RND','MKT']]
Y = A[['PROFIT']]

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X,Y)

pkl.dump(model,open("model.pkl","wb"))
model = pkl.load(open("model.pkl","rb"))
