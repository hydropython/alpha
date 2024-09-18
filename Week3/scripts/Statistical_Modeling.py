import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import shap
import matplotlib.pyplot as plt
import os

class StatisticalModeling:
   
    def __init__(self, file_path):
        # Load the data
        self.data = pd.read_csv(file_path)
        
    def preprocess_data(self, columns):
        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing in the DataFrame: {missing_columns}")

        # Extract specified columns
        self.data = self.data[columns]

        # Handle missing values in categorical columns
        categorical_columns = [col for col in columns if self.data[col].dtype == 'object']
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.data[categorical_columns] = categorical_imputer.fit_transform(self.data[categorical_columns])
        
        # Encode categorical columns
        self.data = pd.get_dummies(self.data, columns=categorical_columns)
        print("Data preprocessing completed.")
        print(self.data.head())

    def split_data(self, target_column, columns):
        # Verify target column exists
        if target_column not in self.data.columns:
            raise KeyError(f"'{target_column}' not found in DataFrame columns.")
        
        # Verify feature columns exist
        missing_cols = [col for col in columns if col not in self.data.columns]
        if missing_cols:
            raise KeyError(f"Missing columns in DataFrame: {', '.join(missing_cols)}")
        
        # Separate features and target variable
        X = self.data[columns]
        y = self.data[target_column]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test):
        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Print the results
        print(f"RMSE: {rmse}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        return rmse, mse, mae

    def run_models(self, X_train, y_train, X_test, y_test):
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        
        for name, model in models.items():
            print(f"\nRunning {name} without hyperparameter tuning...")
            model.fit(X_train, y_train)
            print(f"Evaluation metrics for {name}:")
            self.evaluate_model(model, X_test, y_test)

    def hyperparameter_tuning(self, model_name, X_train, y_train, X_test, y_test):
        print(f"\nPerforming hyperparameter tuning for {model_name} using RandomizedSearchCV...")
        
        if model_name == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
            param_dist = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingRegressor(random_state=42)
            param_dist = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(random_state=42)
            param_dist = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'colsample_bytree': [0.3, 0.7, 1.0]
            }
        elif model_name == 'Decision Tree':
            model = DecisionTreeRegressor(random_state=42)
            param_dist = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3, verbose=1, random_state=42)
        random_search.fit(X_train, y_train)

        # Print the best parameters found by RandomizedSearchCV
        print(f"Best Hyperparameters for {model_name}: {random_search.best_params_}")

        # Evaluate the best model
        print(f"Evaluation metrics for {model_name} after tuning:")
        self.evaluate_model(random_search.best_estimator_, X_test, y_test)

        return random_search.best_estimator_

    def interpret_model(self, model, X_train):
        print("Interpreting the model using SHAP...")

        # Create a SHAP explainer and get SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(X_train)
        
        # Ensure the images directory exists
        import os
        image_dir = "../images"
        os.makedirs(image_dir, exist_ok=True)
        
        # Generate and save SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)  # Prevent showing plot directly
        plt.savefig(f"{image_dir}/shap_summary_plot.png", bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to {image_dir}/shap_summary_plot.png")