import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


s = pd.read_csv("./FinalProject/social_media_usage.csv")

def clean_sm (x):
    x = np.where(x == 1, 1, 0)
    return x
    
toy_df = pd.DataFrame({'A': [5, 1, 12], 'B': [0, 1, 5]})
print(toy_df)

cleaned_toy_df = clean_sm(toy_df)
print(cleaned_toy_df)

ss = s[["income","educ2","par","marital","gender","age","web1h"]]
ss["income"] = np.where(ss["income"] > 9 , np.nan, ss["income"])
ss["educ2"] = np.where(ss["educ2"] > 8 , np.nan, ss["educ2"])
ss["par"] = np.where(ss["par"] == 1 , 1, 0)
ss["marital"] = np.where(ss["marital"] == 1 , 1, 0)
ss["gender"] = np.where(ss["gender"] == 2 , 1, 0)
ss["age"] = np.where(ss["age"] > 98 , np.nan, ss["age"])
ss["web1h"] = clean_sm(ss["web1h"])

ss = ss.rename(columns={"educ2" : "education", "par" : "parent", "marital" : "married", "gender" : "female", "web1h" : "sm_li"})
ss = ss.dropna()

y = ss["sm_li"]
X = ss[["education","income","parent","married","female","age"]]

X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=534)


lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
confusion_matrix(y_test,y_pred)


st.markdown("# Predicting LinkedIn Users")
st.markdown("## Based on their Demographic Data")