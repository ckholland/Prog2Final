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


st.markdown("# Predicting LinkedIn Users")
st.markdown("#### Based on their Demographic Data")

education = st.number_input("Education Level:", min_value=1, max_value=8, step=1)
income = st.number_input("Income:", min_value=1, max_value=9, step=1)
parent = st.number_input("Parent:", min_value=0, max_value=120, step=1)
married = st.number_input("Married:", min_value=0, max_value=1, step=1)
female = st.number_input("Female:", min_value=0, max_value=1, step=1)
age = st.number_input("Age:", min_value=0, max_value=98, step=1)

person = [education, income, parent, married, female, age]

probability = lr.predict_proba([person])
prediction = np.where(lr.predict([person]) == 1, "LinkedIn User", "Not a LinkedIn User")

st.write(probability[0][1])
st.write(prediction[0][0])


## Tomorrow I need to make it look pretty, mix up the input types and submit. I think it should be colored based on the success in the output