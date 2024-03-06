# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

available_features = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)',
]

def train_knn_iris(features: list[str], n_neighbors=3) -> float:
    """Train and evaluate a KNN model on the Iris dataset.

    Train a KNN model on the Iris dataset with user-selected features and number of neighbors.
    
    Parameters
    ----------
    features : list[str]
        List of feature names to use (e.g., ['sepal length (cm)', 'sepal width (cm)']).
    n_neighbors : int, default 3
        Number of neighbors to use for KNN.
    
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
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Train the model using the training sets
    knn.fit(X_train, y_train)
    
    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    
    # Model Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with features {features} and {n_neighbors} neighbors: {accuracy}")
    return accuracy
