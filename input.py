import argparse

parser = argparse.ArgumentParser()

parser.add_argument("text" , type=str)
args = parser.parse_args()

import joblib
import pandas as pd

model1 = joblib.load("joblib_model.pkl")
vec = joblib.load("joblib_v.pkl")

text_d = pd.Series(args.text)
text_dt = vec.transform(text_d)

prd = model1.predict(text_dt)

if prd[0] == 1:
    print("this tweet is real")
else:
    print("this tweet is fake")