import joblib

# Load the model
model_path = 'D:/bug sev v2 p3/bug sev v2/models/saved/Best_Stacking_Model.pkl'
model = joblib.load(model_path)

# Function to get the number of features from the model
def get_num_features(model):
    # Check the type of model and extract the number of features accordingly
    if hasattr(model, 'named_estimators_'):
        # For StackingClassifier, get the features from one of the base models
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'get_booster'):
                return estimator.get_booster().num_features()
    return None

# Get the number of features
num_features = get_num_features(model)
print(f"Number of features: {num_features}")
