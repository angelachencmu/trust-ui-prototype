# Trust UI Prototype

This is a prototype of the Trust UI project.
It is a simple web application that aims to collect telemetry data from users to understand the trust between humans and machines.

This repository is developed using Python 3.11.8, but it _should_ work with any Python 3.8+ version.

## How to run

This web application uses [Streamlit](https://streamlit.io/) to run.

First, create a virtual environment and install the dependencies:
```bash
python -m venv ./env
source env/bin/activate
pip install -r requirements.txt
```

Then, run the application:
```bash
streamlit run deployment/test.py
```
# test.py

This script creates a web application designed for interacting with and evaluating machine learning models using either the Breast Cancer or Iris datasets. Here's a breakdown of its functionality and structure:

## Title and Dataset Selection:
It begins by setting a title for the app and offering a choice between two datasets: Breast Cancer and Iris, using a selectbox for selection.
## Data Preparation:
Depending on the chosen dataset, it loads the data and creates a pandas DataFrame with features and target labels. It also provides an option to display this DataFrame in the app.
## Machine Learning Model Selection and Configuration:
The user can enter a unique ID, which is used for tracking purposes.
There's functionality to select specific features from the dataset to include in the training process.
The user can choose between three machine learning algorithms: Decision Tree, K-Nearest Neighbors (KNN), and Random Forest. Each algorithm has its own set of parameters that can be adjusted through the sidebar.
## Training and Evaluation:
Upon selecting the desired features and algorithm configurations, the user can train the model. The training process includes splitting the data into training and test sets, fitting the model on the training data, and then evaluating it on the test set.
For the Decision Tree algorithm, it additionally visualizes the trained decision tree.
## Interaction Logging and Export:
The script logs each interaction, including the start and end time of the model training, the selected features, the dataset, the chosen algorithm, and the achieved accuracy.
It stores these logs in a session state and provides an option to display them in the app.
Users can download their interaction logs as a CSV file.
## Repeatable Process:
The script seems to be designed for repeatable use, allowing for multiple interactions and model trainings during a session. This is facilitated through unique keys for Streamlit widgets and session state management for logging interactions.
This app offers an interactive tool for users to experiment with different machine learning models and parameters, providing immediate feedback on model performance and the ability to compare results across multiple runs.
