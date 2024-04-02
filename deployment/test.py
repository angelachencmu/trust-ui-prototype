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
    with open('interaction_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start Time', 'End Time', 'Duration', 'Selected Features'])
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

while True:
    # Allow users to choose features to train
    selected_features = st.multiselect('Select features to train', breast_cancer.feature_names)

    if len(selected_features) > 0:
        start_time = time.time()  # Record the start time

        features = df[selected_features].values
        labels = df['diagnosis'].values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

        alg = ['Decision Tree', 'K-Nearest Neighbors', 'Random Forest']
        classifier = st.selectbox('Which algorithm?', alg)

        # Allow users to hide the sidebar
        show_sidebar = st.checkbox('Show sidebar', value=True)

        if show_sidebar:
            # ... (rest of the code for each classifier)

        else:
            # ... (rest of the code for each classifier)

        end_time = time.time()  # Record the end time
        duration = end_time - start_time
        interactions.append([start_time, end_time, duration, ','.join(selected_features)])  # Store the interaction

    else:
        st.write("Please select at least one feature to train the models.")
        break

    if st.button('Finish'):
        break

log_interactions(interactions)  # Write all interactions to the CSV file

# Display the interaction log as a table
if len(interactions) > 0:
    st.subheader('Interaction Log')
    log_df = pd.DataFrame(interactions, columns=['Start Time', 'End Time', 'Duration', 'Selected Features'])
    st.table(log_df)
else:
    st.write("No interactions recorded.")
