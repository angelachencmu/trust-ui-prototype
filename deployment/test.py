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
import os

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

# User identification
user_id = st.text_input('Enter your user ID:')

interactions = []
iteration_counter = 0

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
        # ... (hyperparameter selection code remains the same)

    else:
        # ... (default hyperparameter code remains the same)

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
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        interactions.append([user_id, start_time_str, end_time_str, duration, ','.join(selected_features), classifier, acc])  # Store the interaction with user ID, start time, end time, accuracy, and algorithm type
        st.write('Accuracy: ', acc)

        # Write interactions to CSV file
        csv_file = f"{user_id}_interactions.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['User ID', 'Start Time', 'End Time', 'Duration (seconds)', 'Selected Features', 'Algorithm', 'Accuracy'])
            writer.writerows(interactions)

# Display the current interaction log as a table
if len(interactions) > 0:
    st.subheader('Current Interaction Log')
    log_df = pd.DataFrame(interactions, columns=['User ID', 'Start Time', 'End Time', 'Duration (seconds)', 'Selected Features', 'Algorithm', 'Accuracy'])
    st.table(log_df)
else:
    st.write("No interactions recorded.")
