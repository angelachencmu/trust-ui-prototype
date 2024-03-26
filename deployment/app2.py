# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

available_features = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)',
]

def train_decision_tree_iris(features: list[str]) -> float:
    """Train and evaluate a decision tree model on the Iris dataset.

    Train a decision tree model on the Iris dataset with user-selected features.

    Parameters
    ----------
    features : list[str]
        List of feature names to use (e.g., ['sepal length (cm)', 'sepal width (cm)']).

    Returns
    -------
    accuracy : float
        The accuracy of the model on the test set.
    """
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, [iris.feature_names.index(feature) for feature in features]]
    y = iris.target

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create decision tree model
    dt = DecisionTreeClassifier()

    # Train the model using the training sets
    dt.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = dt.predict(X_test)

    # Model Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with features {features}: {accuracy}")
    return accuracy
