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

education = st.selectbox("Which of the following best describes their eduction level?",
			options = ["Never went to high school", 
              "Never graduated high school", 
              "Graduated high school or GED",
              "Attended some college",
              "Two-year associate degree",
              "Four-year bachelor's degree",
              "Some post-graduate or professional schooling",
              "Postgraduate or professional degree including master's"])
if education == "Never went to high school":
    education = 1
elif education == "Never graduated high school":
    education = 2
elif education == "Graduated high school or GED":
    education = 3
elif education == "Attended some college":
    education = 4
elif education == "Two-year associate degree":
    education = 5
elif education == "Four-year bachelor's degree":
    education = 6
elif education == "Some post-graduate or professional schooling":
    education = 7
elif education == "Postgraduate or professional degree including master's":
    education = 8

income_options = {
    "Less than $10,000": 1,
    "$10,000 to under $20,000": 2,
    "$20,000 to under $30,000": 3,
    "$30,000 to under $40,000": 4,
    "$40,000 to under $50,000": 5,
    "$50,000 to under $75,000": 6,
    "$75,000 to under $100,000": 7,
    "$100,000 to under $150,000": 8,
    "$150,000 or more": 9,
    }
income = st.selectbox("What is their annual household income?",
			income_options.keys())


parent = 0
if st.checkbox("Are they a parent?"):
    parent = 1

married = 0
if st.checkbox("Are they married?"):
    married = 1

female_options = {
    "Male": 0,
    "Female": 1,
    }

female = st.selectbox("What is their sex?",
			female_options.keys())

age = st.number_input("How old are they?", min_value=0, max_value=98, step=1)

person = [education, income_options[income], parent, married, female_options[female], age]

probability = lr.predict_proba([person])
prediction = np.where(lr.predict([person]) == 1, "LinkedIn User", "Not a LinkedIn User")

color = "green" if probability[0][1] >= 0.5 else "red"
# Create an H1 heading with dynamic color
st.markdown(
    f"<h1 style='color: {color};'>{round(probability[0][1] * 100, 2)}%</h1>", unsafe_allow_html=True)

st.markdown(
    f"<h1 style='color: {color};'>{prediction[0]}</h1>", unsafe_allow_html=True)

if prediction == "LinkedIn User":
    st.image("./LILogo.png", width = 100)




## Tomorrow I need to make it look pretty, mix up the input types and submit. I think it should be colored based on the success in the output, also should make one of those guages to show probability. That should be enough.