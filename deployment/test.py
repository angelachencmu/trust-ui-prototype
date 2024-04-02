import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import csv

def log_interactions(interactions):
    with open('interaction_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(interactions)

st.title('Breast Cancer Classifier')

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()

# Create a DataFrame
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df['target'] = breast_cancer.target
df['diagnosis'] = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)

if st.checkbox('Show dataframe'):
    st.write(df)

st.subheader('Machine Learning Models')

interactions = []
iteration_counter = 0

while True:
    # Allow users to choose features to train
    selected_features = st.multiselect('Select features to train', breast_cancer.feature_names, key=f'feature_selection_{iteration_counter}')

    if len(selected_features) > 0:
        start_time = time.time()  # Record the start time

        features = df[selected_features].values
        labels = df['diagnosis'].values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

        alg = ['Decision Tree', 'K-Nearest Neighbors', 'Random Forest']
        classifier = st.selectbox('Which algorithm?', alg, key=f'classifier_{iteration_counter}')

        # Allow users to show/hide hyperparameters
        show_hyperparameters = st.checkbox('Show hyperparameters', value=True, key=f'show_hyperparameters_{iteration_counter}')

        if show_hyperparameters:
            if classifier == 'Decision Tree':
                st.sidebar.markdown("# Decision Tree Classifier")
                criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'), key=f'dt_criterion_{iteration_counter}',
                                                 help="Function to measure the quality of a split. 'gini' for the Gini impurity and 'entropy' for the information gain.")
                splitter = st.sidebar.selectbox('Splitter', ('best', 'random'), key=f'dt_splitter_{iteration_counter}',
                                                help="Strategy used to choose the split at each node. 'best' selects the best split, while 'random' selects the best random split.")
                max_depth = int(st.sidebar.number_input('Max Depth', key=f'dt_max_depth_{iteration_counter}',
                                                        help="Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples."))
                min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2, key=f'dt_min_samples_split_{iteration_counter}',
                                                      help="Minimum number of samples required to split an internal node.")
                min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=f'dt_min_samples_leaf_{iteration_counter}',
                                                     help="Minimum number of samples required to be at a leaf node.")
                max_features = st.sidebar.slider('Max Features', 1, len(selected_features), len(selected_features), key=f'dt_max_features_{iteration_counter}',
                                                 help="Number of features to consider when looking for the best split. If 'auto', sqrt(n_features) is used.")
                max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes', key=f'dt_max_leaf_nodes_{iteration_counter}',
                                                             help="Maximum number of leaf nodes. If None, unlimited number of leaf nodes."))
                min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease', key=f'dt_min_impurity_decrease_{iteration_counter}',
                                                                help="A node will be split if this split induces a decrease of the impurity greater than or equal to this value.")

                if max_depth == 0:
                    max_depth = None
                if max_leaf_nodes == 0:
                    max_leaf_nodes = None

                dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=42,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                             min_impurity_decrease=min_impurity_decrease)

            elif classifier == 'K-Nearest Neighbors':
                st.sidebar.markdown("# K-Nearest Neighbors")
                n_neighbors = st.sidebar.slider('Number of Neighbors (n_neighbors)', 1, 15, 5, key=f'knn_n_neighbors_{iteration_counter}',
                                                help="Number of neighbors to use for classification.")
                weights = st.sidebar.selectbox('Weights', ('uniform', 'distance'), key=f'knn_weights_{iteration_counter}',
                                               help="Weight function used in prediction. 'uniform' assigns equal weights to all neighbors, while 'distance' assigns weights proportional to the inverse of the distance.")
                algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'), key=f'knn_algorithm_{iteration_counter}',
                                                 help="Algorithm used to compute the nearest neighbors.")
                leaf_size = st.sidebar.slider('Leaf Size', 1, 50, 30, key=f'knn_leaf_size_{iteration_counter}',
                                              help="Leaf size passed to BallTree or KDTree algorithms.")
                p = st.sidebar.slider('p (Power Parameter)', 1, 5, 2, key=f'knn_p_{iteration_counter}',
                                      help="Power parameter for the Minkowski metric.")
                metric = st.sidebar.selectbox('Metric', ('euclidean', 'manhattan', 'minkowski', 'chebyshev'), key=f'knn_metric_{iteration_counter}',
                                              help="Distance metric to use for the tree.")
                n_jobs = st.sidebar.slider('Number of Jobs (n_jobs)', -1, 4, 1, key=f'knn_n_jobs_{iteration_counter}',
                                           help="Number of parallel jobs to run for neighbors search. -1 uses all available processors.")

                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                           leaf_size=leaf_size, p=p, metric=metric, n_jobs=n_jobs)

            elif classifier == 'Random Forest':
                st.sidebar.markdown("# Random Forest Classifier")
                n_estimators = st.sidebar.slider('Number of Estimators (n_estimators)', 1, 200, 100, key=f'rf_n_estimators_{iteration_counter}',
                                                 help="Number of trees in the forest.")
                criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy'), key=f'rf_criterion_{iteration_counter}',
                                                 help="Function to measure the quality of a split. 'gini' for the Gini impurity and 'entropy' for the information gain.")
                max_depth = int(st.sidebar.number_input('Max Depth', key=f'rf_max_depth_{iteration_counter}',
                                                        help="Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples."))
                min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2, key=f'rf_min_samples_split_{iteration_counter}',
                                                      help="Minimum number of samples required to split an internal node.")
                min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1, key=f'rf_min_samples_leaf_{iteration_counter}',
                                                     help="Minimum number of samples required to be at a leaf node.")
                max_features = st.sidebar.selectbox('Max Features', ('sqrt', 'log2'), key=f'rf_max_features_{iteration_counter}',
                                                    help="Number of features to consider when looking for the best split.")
                bootstrap = st.sidebar.selectbox('Bootstrap', (True, False), key=f'rf_bootstrap_{iteration_counter}',
                                                 help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
                random_state = int(st.sidebar.number_input('Random State', value=42, key=f'rf_random_state_{iteration_counter}',
                                                           help="Controls the randomness of the bootstrapping of the samples used when building trees."))

                if max_depth == 0:
                    max_depth = None

                rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features, bootstrap=bootstrap, random_state=random_state)

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

            elif classifier == 'Random Forest':
                n_estimators = 100
                criterion = 'gini'
                max_depth = None
                min_samples_split = 2
                min_samples_leaf = 1
                max_features = 'sqrt'
                bootstrap = True
                random_state = 42

                rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             max_features=max_features, bootstrap=bootstrap, random_state=random_state)

        if st.button('Train Model', key=f'train_model_{iteration_counter}'):
            if classifier == 'Decision Tree':
                dtc.fit(X_train, y_train)
                acc = dtc.score(X_test, y_test)
                tree = export_graphviz(dtc, feature_names=selected_features)
                st.graphviz_chart(tree)
            elif classifier == 'K-Nearest Neighbors':
                knn.fit(X_train, y_train)
                acc = knn.score(X_test, y_test)
            elif classifier == 'Random Forest':
                rfc.fit(X_train, y_train)
                acc = rfc.score(X_test, y_test)

            end_time = time.time()  # Record the end time
            duration = round(end_time - start_time, 2)
            interactions.append([duration, ','.join(selected_features), classifier, acc])  # Store the interaction with accuracy and algorithm type
            st.write('Accuracy: ', acc)
        else:
            break
    else:
        break

    iteration_counter += 1

# Display the interaction log as a table
if len(interactions) > 0:
    st.subheader('Interaction Log')
    log_df = pd.DataFrame(interactions, columns=['Duration (seconds)', 'Selected Features', 'Algorithm', 'Accuracy'])
    st.table(log_df)
    log_interactions(interactions)  # Append interactions to the CSV file
else:
    st.write("No interactions recorded.")
