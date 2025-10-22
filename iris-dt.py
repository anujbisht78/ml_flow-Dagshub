import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='anujbisht78', repo_name='ml_flow-Dagshub', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/anujbisht78/ml_flow-Dagshub.mlflow")


# Load the Iris dataset
iris=load_iris()
X=iris.data
y=iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the RandomForestClassifier
max_depth = 5

# Setting up MLflow experiment name
mlflow.set_experiment("iris-dt")



# Start an MLflow run
with mlflow.start_run():
    
    # Initialize the RandomForestClassifier model
    model=DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Tracking parameters and metrics with MLflow
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    
    # Tracking the plots with MLflow
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Save the plot to a file
    plt.savefig("confusion_matrix.png")
    
    # Log the confusion matrix plot as an artifact
    mlflow.log_artifact("confusion_matrix.png")
    
    # log the code
    mlflow.log_artifact(__file__)
    
    # Save model locally if you still want to keep it
    import joblib
    joblib.dump(model, "decision_tree_model.pkl")

    # Log model path info as a string (not the model file)
    mlflow.log_param("model_path", "decision_tree_model.pkl")
    
    # # log the model
    # mlflow.sklearn.log_model(model, "decision_tree_model")
    
    print(f"Model accuracy: {accuracy}")