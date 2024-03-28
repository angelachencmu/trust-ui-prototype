import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import load_iris

st.title('Iris')

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['variety'] = pd.Categorical.from_codes(iris.target, iris.target_names)

if st.checkbox('Show dataframe'):
    st.write(df)

st.subheader('Histogram')
feature = st.selectbox('Which feature?', df.columns[0:4])
species = st.multiselect('Show iris per variety?', df['variety'].unique())

# Filter dataframe
new_df2 = df[(df['variety'].isin(species))][feature]
fig2 = px.histogram(new_df2, x=feature, color="variety", marginal="rug")
st.plotly_chart(fig2)

st.subheader('Machine Learning models')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

features = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
labels = df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

alg = ['Decision Tree', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)

if classifier == 'Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc = confusion_matrix(y_test, pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)
elif classifier == 'Support Vector Machine':
    svm = SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm = confusion_matrix(y_test, pred_svm)
    st.write('Confusion matrix: ', cm)
