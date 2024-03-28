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
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier

st.title('Iris')

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['variety'] = pd.Categorical.from_codes(iris.target, iris.target_names)

if st.checkbox('Show dataframe'):
    st.write(df)

# Allow users to hide the histogram section
show_histogram = st.checkbox('Show histogram', value=True)

if show_histogram:
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

# Allow users to choose features to train
selected_features = st.multiselect('Select features to train', iris.feature_names)

if len(selected_features) > 0:
    features = df[selected_features].values
    labels = df['variety'].values

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

    alg = ['Decision Tree', 'K-Nearest Neighbors']
    classifier = st.selectbox('Which algorithm?', alg)

    # Allow users to hide the sidebar
    show_sidebar = st.checkbox('Show sidebar', value=True)

    if show_sidebar:
        if classifier == 'Decision Tree':
            st.sidebar.markdown("# Decision Tree Classifier")
            criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'),
                                             help="Function to measure the quality of a split. 'gini' for the Gini impurity and 'entropy' for the information gain.")
            splitter = st.sidebar.selectbox('Splitter', ('best', 'random'),
                                            help="Strategy used to choose the split at each node. 'best' selects the best split, while 'random' selects the best random split.")
            max_depth = int(st.sidebar.number_input('Max Depth',
                                                    help="Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples."))
            min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2, key=1234,
                                                  help="Minimum number of samples required to split an internal node.")
            min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=1235,
                                                 help="Minimum number of samples required to be at a leaf node.")
            max_features = st.sidebar.slider('Max Features', 1, len(selected_features), len(selected_features), key=1236,
                                             help="Number of features to consider when looking for the best split. If 'auto', sqrt(n_features) is used.")
            max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes',
                                                         help="Maximum number of leaf nodes. If None, unlimited number of leaf nodes."))
            min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease',
                                                            help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.")

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
            tree = export_graphviz(dtc, feature_names=selected_features)
            st.graphviz_chart(tree)

        elif classifier == 'K-Nearest Neighbors':
            st.sidebar.markdown("# K-Nearest Neighbors")
            n_neighbors = st.sidebar.slider('Number of Neighbors (n_neighbors)', 1, 15, 5, key=1237,
                                            help="Number of neighbors to use for classification.")
            weights = st.sidebar.selectbox('Weights', ('uniform', 'distance'),
                                           help="Weight function used in prediction. 'uniform' assigns equal weights to all neighbors, while 'distance' assigns weights proportional to the inverse of the distance.")
            algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'),
                                             help="Algorithm used to compute the nearest neighbors.")
            leaf_size = st.sidebar.slider('Leaf Size', 1, 50, 30, key=1238,
                                          help="Leaf size passed to BallTree or KDTree algorithms.")
            p = st.sidebar.slider('p (Power Parameter)', 1, 5, 2, key=1239,
                                  help="Power parameter for the Minkowski metric.")
            metric = st.sidebar.selectbox('Metric', ('euclidean', 'manhattan', 'minkowski', 'chebyshev'),
                                          help="Distance metric to use for the tree.")
            n_jobs = st.sidebar.slider('Number of Jobs (n_jobs)', -1, 4, 1, key=1240,
                                       help="Number of parallel jobs to run for neighbors search. -1 uses all available processors.")

            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size, p=p, metric=metric, n_jobs=n_jobs)
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            st.write('Accuracy: ', acc)
    else:
        if classifier == 'Decision Tree':
            criterion = 'gini'
            splitter = 'best'
            max_depth = None
            min_samples_split = 2
            min_samples_leaf = 1
            max_features = len(selected_features)
            max_leaf_nodes = None
            min_impurity_decrease = 0.0

            dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=42,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                         min_impurity_decrease=min_impurity_decrease)
            dtc.fit(X_train, y_train)
            acc = dtc.score(X_test, y_test)
            st.write('Accuracy: ', acc)
            tree = export_graphviz(dtc, feature_names=selected_features)
            st.graphviz_chart(tree)

        elif classifier == 'K-Nearest Neighbors':
            n_neighbors = 5
            weights = 'uniform'
            algorithm = 'auto'
            leaf_size = 30
            p = 2
            metric = 'euclidean'
            n_jobs = 1

            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size, p=p, metric=metric, n_jobs=n_jobs)
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            st.write('Accuracy: ', acc)
else:
    st.write("Please select at least one feature to train the models.")
