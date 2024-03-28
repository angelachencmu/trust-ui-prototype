import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import export_graphviz

st.title('Iris Dataset Analysis')

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

if not new_df2.empty:
    if 'variety' in df.columns:
        fig2 = px.histogram(df[(df['variety'].isin(species))], x=feature, color="variety", marginal="rug")
    else:
        fig2 = px.histogram(new_df2, x=feature, marginal="rug")
    st.plotly_chart(fig2)
else:
    st.write("No data selected.")

st.subheader('Machine Learning Models')

features = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
labels = df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

alg = ['Decision Tree', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)

if classifier == 'Decision Tree':
    criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
    splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))
    max_depth = st.sidebar.slider('Max Depth', 1, 50, 5)
    min_samples_split = st.sidebar.slider('Min Samples Split', 1, 10, 2)
    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, 10, 1)

    dtc = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=1
    )
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    y_pred = dtc.predict(X_test)
    cm_dtc = confusion_matrix(y_test, y_pred)
    st.write('Confusion matrix: ', cm_dtc)

    # Plot the decision tree
    fig_tree, ax_tree = plt.subplots(figsize=(15, 10))
    export_graphviz(dtc, out_file=None, filled=True, rounded=True, feature_names=iris.feature_names, class_names=iris.target_names)
    st.pyplot(fig_tree)

elif classifier == 'Support Vector Machine':
    kernel = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
    C = st.sidebar.slider('C', 0.1, 10.0, 1.0)
    gamma = st.sidebar.selectbox('Gamma', ('scale', 'auto'))

    svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=1)
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    y_pred = svm.predict(X_test)
    cm_svm = confusion_matrix(y_test, y_pred)
    st.write('Confusion matrix: ', cm_svm)
