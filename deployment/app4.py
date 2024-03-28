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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import export_graphviz

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

if not new_df2.empty:
    if 'variety' in df.columns:
        fig2 = px.histogram(df[(df['variety'].isin(species))], x=feature, color="variety", marginal="rug")
    else:
        fig2 = px.histogram(new_df2, x=feature, marginal="rug")
    st.plotly_chart(fig2)
else:
    st.write("No data selected.")

st.subheader('Machine Learning models')

features = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
labels = df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

alg = ['Decision Tree', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)

if classifier == 'Decision Tree':
    st.sidebar.markdown("# Decision Tree Classifier")
    criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'))
    splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))
    max_depth = int(st.sidebar.number_input('Max Depth'))
    min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2, key=1234)
    min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=1235)
    max_features = st.sidebar.slider('Max Features', 1, 4, 4, key=1236)
    max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))
    min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

    if max_depth == 0:
        max_depth = None
    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=42,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                 min_impurity_decrease=min_impurity_decrease)
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc = confusion_matrix(y_test, pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)
    tree = export_graphviz(dtc, feature_names=iris.feature_names)
    st.graphviz_chart(tree)

elif classifier == 'Support Vector Machine':
    svm = SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm = confusion_matrix(y_test, pred_svm)
    st.write('Confusion matrix: ', cm)
